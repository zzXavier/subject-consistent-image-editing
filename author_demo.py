"""Run the original Tencent IPAdapter.generate API on a real held-out catalog reference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from src.data import ROOT, items, prepare
os.environ.setdefault("HF_HOME",str(ROOT/".cache"/"huggingface"))

PIN = "62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True,help="pinned tencent-ailab/IP-Adapter checkout")
    parser.add_argument("--output",type=Path,default=ROOT/"work"/"author")
    args=parser.parse_args()
    sha=subprocess.check_output(["git","-C",str(args.source),"rev-parse","HEAD"],text=True).strip()
    if sha!=PIN or subprocess.check_output(["git","-C",str(args.source),"status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("Expected a clean pinned author checkout")
    sys.path.insert(0,str(args.source.resolve()))
    import torch
    from diffusers import StableDiffusionPipeline,DDIMScheduler
    from ip_adapter import IPAdapter
    from src.assets import models
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the author's fp16 implementation")
    base,ip=models()
    pipe=StableDiffusionPipeline.from_pretrained(str(base),variant="fp16",torch_dtype=torch.float16,
            safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True)
    # Same DDIM schedule used in the upstream SD1.5 demonstration.
    pipe.scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,
            beta_schedule="scaled_linear",clip_sample=False,set_alpha_to_one=False,steps_offset=1)
    model=IPAdapter(pipe,str(ip/"models/image_encoder"),str(ip/"models/ip-adapter_sd15.safetensors"),"cuda")
    item=next(i for i in items() if i["split"]=="test" and i["id"]!="B083V9C8D2")
    reference,_,source_sha=prepare(item,ref=True)
    prompt=f"product photo of a {item['name']} package on a blue studio background"
    image=model.generate(pil_image=reference,prompt=prompt,num_samples=1,num_inference_steps=12,
                         seed=13,scale=.6,width=384,height=384)[0]
    args.output.mkdir(parents=True,exist_ok=True)
    image.save(args.output/"generated.png")
    reference.save(args.output/"reference.png")
    details={"upstream_commit":sha,"api":"ip_adapter.IPAdapter.generate","subject":item["id"],
             "reference_sha256":source_sha,"prompt":prompt,"steps":12,"seed":13,"scale":.6,
             "scope":"reference-only text-to-image example, separate from paired inpainting metrics"}
    (args.output/"run.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    print(args.output.resolve())


if __name__=="__main__":
    main()
