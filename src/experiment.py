from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageOps
from src.data import ROOT, items, prepare

SD = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
SD_REV = '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb'
IP = 'h94/IP-Adapter'
IP_REV = '018e402774aeeddd60609b4ecdb7e298259dc729'

def pixel_scores(original, result, mask, color):
    x=np.asarray(original,dtype=float); y=np.asarray(result,dtype=float); keep=np.asarray(mask)>127
    if not keep.any() or keep.all(): raise ValueError('Foreground and editable background must both be present')
    axis={'red':0,'blue':2}[color]
    dominant=(y[:,:,axis]-np.delete(y,axis,2).max(axis=2))>16
    return {'outside_edit_mae':float(np.abs(x-y).mean(axis=2)[keep].mean()),
            'requested_hue_fraction':float(dominant[~keep].mean())}

def evaluate(out):
    rows=[]
    run=json.loads((out/'manifest.json').read_text())
    for case in run['cases']:
        folder=out/case['case']; source=Image.open(folder/'source.png'); mask=Image.open(folder/'mask.png')
        for name,score in case['reference_cosine'].items():
            edited=Image.open(folder/(name+'.png'))
            rows.append(dict(subject=case['id'],split=case['split'],color=case['color'],method=name,
                             reference_cosine=score,**pixel_scores(source,edited,mask,case['color'])))
    with (out/'scores.csv').open('w',newline='') as outcsv:
        w=csv.DictWriter(outcsv,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for m in ('text','reference','regional'):
        group=[r for r in rows if r['split']=='test' and r['method']==m]
        if group:
            summary[m]={k:float(np.mean([r[k] for r in group])) for k in ('reference_cosine','outside_edit_mae','requested_hue_fraction')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

def run(out,demo=False):
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.image_processor import IPAdapterMaskProcessor
    from huggingface_hub import snapshot_download
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    assets=snapshot_download(IP,revision=IP_REV,allow_patterns=['models/image_encoder/config.json','models/image_encoder/model.safetensors','models/ip-adapter_sd15.safetensors'])
    encoder=CLIPVisionModelWithProjection.from_pretrained(Path(assets)/'models/image_encoder',torch_dtype=torch.float16)
    pipe=StableDiffusionInpaintPipeline.from_pretrained(SD,revision=SD_REV,variant='fp16',torch_dtype=torch.float16,image_encoder=encoder)
    pipe.load_ip_adapter(assets,subfolder='models',weight_name='ip-adapter_sd15.safetensors')
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    prep=CLIPImageProcessor()
    def encode(image):
        tensor=prep(images=image,return_tensors='pt').pixel_values.to('cuda',dtype=torch.float16)
        with torch.inference_mode():
            feat=pipe.image_encoder(tensor).image_embeds.float().cpu()
        return torch.nn.functional.normalize(feat)
    out.mkdir(parents=True,exist_ok=True)
    manifest={'model':SD_REV,'adapter':IP_REV,'steps':12,'size':384,'seed':13,'cases':[]}
    selected=[case for case in items() if case['id']!='B083V9C8D2']
    for item in (selected[:1] if demo else selected):
        source,mask,_=prepare(item); reference,_,_=prepare(item,ref=True)
        empty=Image.new('RGB',source.size,'white')
        ref_embedding=encode(reference)
        for color in (['red'] if demo else ['red','blue']):
            key=item['id']+'-'+color; sub=out/key;sub.mkdir(exist_ok=True)
            source.save(sub/'source.png'); reference.save(sub/'reference.png'); mask.save(sub/'mask.png')
            cosines={}
            for name,scale in (('text',0.0),('reference',0.6),('regional',0.6)):
                pipe.set_ip_adapter_scale(scale)
                args={}
                if name=='regional':
                    processed=IPAdapterMaskProcessor().preprocess(mask,height=384,width=384)
                    args['cross_attention_kwargs']={'ip_adapter_masks':[processed.reshape(1,1,384,384)]}
                image=pipe(prompt=f"A photo of the same {item['name']} on a plain {color} studio background",
                           image=source,mask_image=ImageOps.invert(mask),ip_adapter_image=reference,
                           height=384,width=384,num_inference_steps=12,generator=torch.Generator('cuda').manual_seed(13),**args).images[0]
                image.save(sub/(name+'.png'))
                cosines[name]=float((ref_embedding*encode(Image.composite(image,empty,mask))).sum())
            manifest['cases'].append({'case':key,'id':item['id'],'split':item['split'],'color':color,'reference_cosine':cosines})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluate(out)

