from pathlib import Path
import csv
import json
import platform
from importlib.metadata import version
import numpy as np
from time import perf_counter
from PIL import Image, ImageOps
from src.data import ROOT, items, prepare
from src.assets import IP_REV, SD_REV


def pixel_scores(original, result, mask, color):
    x = np.asarray(original, dtype=float)
    y = np.asarray(result, dtype=float)
    keep = np.asarray(mask) > 127
    if not keep.any() or keep.all():
        raise ValueError("Foreground and editable background must both be present")
    axis = {"red": 0, "blue": 2}[color]
    dominant = (y[:, :, axis] - np.delete(y, axis, 2).max(axis=2)) > 16
    return {
        "outside_edit_mae": float(np.abs(x - y).mean(axis=2)[keep].mean()),
        "requested_hue_fraction": float(dominant[~keep].mean()),
    }


def subject_signature(image, mask):
    pixels = np.asarray(image.convert("RGB"))[np.asarray(mask)>127]
    if not len(pixels):
        raise ValueError("Empty foreground in color retrieval")
    hist = np.histogramdd(pixels,bins=(6,6,6),range=[(0,256)]*3)[0]
    return np.sqrt(hist.ravel()/len(pixels))


def write_manifest(out, manifest):
    partial = out/"manifest.tmp"
    partial.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    partial.replace(out/"manifest.json")


def evaluate(out):
    rows = []
    run = json.loads((out / "manifest.json").read_text())
    if (
        run["status"] != "completed"
        or len({c["case"] for c in run["cases"]}) != run["expected_cases"]
    ):
        raise ValueError("Incomplete inference manifest")
    gallery = {}
    for case in run["cases"]:
        if case["id"] not in gallery:
            folder = out/case["case"]
            with Image.open(folder/"reference.png") as reference, Image.open(folder/"reference-mask.png") as mask:
                gallery[case["id"]] = subject_signature(reference,mask)
    source_gallery = {}
    for case in run["cases"]:
        folder = out / case["case"]
        source = Image.open(folder / "source.png")
        mask = Image.open(folder / "mask.png")
        if len(gallery)>1 and case["id"] not in source_gallery:
            sig=subject_signature(source,mask)
            sims={key:float(sig @ reference) for key,reference in gallery.items()}
            margin=sims[case["id"]]-max(val for key,val in sims.items() if key!=case["id"])
            source_gallery[case["id"]]={"split":case["split"],"correct":margin>0,"margin":margin}
        features = np.load(folder / "identity_embeddings.npz")
        for name in ("text", "reference", "regional"):
            score = float(np.dot(features["reference_subject"], features[name]))
            edited = Image.open(
                folder
                / (("reference-edited" if name == "reference" else name) + ".png")
            )
            descriptor = subject_signature(edited,mask)
            similarities = {key:float(np.dot(descriptor,sig)) for key,sig in gallery.items()}
            distractors = [val for key,val in similarities.items() if key!=case["id"]]
            margin = similarities[case["id"]]-max(distractors) if distractors else None
            rows.append(
                dict(
                    subject=case["id"],
                    split=case["split"],
                    color=case["color"],
                    method=name,
                    reference_cosine=score,
                    color_gallery_correct=float(margin>0) if margin is not None else None,
                    color_gallery_margin=margin,
                    **pixel_scores(source, edited, mask, case["color"]),
                )
            )
    with (out / "scores.csv").open("w", newline="") as outcsv:
        w = csv.DictWriter(outcsv, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for m in ("text", "reference", "regional"):
        group = [r for r in rows if r["split"] == "test" and r["method"] == m]
        if group:
            summary[m] = {
                k: float(np.mean([r[k] for r in group]))
                for k in (
                    "reference_cosine",
                    "outside_edit_mae",
                    "requested_hue_fraction",
                    "color_gallery_correct",
                    "color_gallery_margin",
                )
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "source-gallery.json").write_text(json.dumps(source_gallery,indent=2))
    from src.panels import panel

    case = next((c for c in run["cases"] if c["split"] == "test"), run["cases"][0])
    panel(out / case["case"], out / "comparison.jpg")
    group = [r for r in rows if r["split"] == "test" and r["method"] == "regional"]
    if group:
        worst = min(group, key=lambda r: r["requested_hue_fraction"])
        panel(
            out / (worst["subject"] + "-" + worst["color"]),
            out / "lowest-scene-score.jpg",
        )


def run(out, demo=False):
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.image_processor import IPAdapterMaskProcessor
    from src.assets import models
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    base, assets = models()
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        Path(assets) / "models/image_encoder", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True
    )
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(base),
        variant="fp16",
        torch_dtype=torch.float16,
        image_encoder=encoder,
        safety_checker=None,
        feature_extractor=CLIPImageProcessor(),
        requires_safety_checker=False,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.load_ip_adapter(
        str(assets), subfolder="models", weight_name="ip-adapter_sd15.safetensors"
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    prep = CLIPImageProcessor()

    def encode(image):
        tensor = prep(images=image, return_tensors="pt").pixel_values.to(
            "cuda", dtype=torch.float16
        )
        with torch.inference_mode():
            feat = pipe.image_encoder(tensor).image_embeds.float().cpu()
        return torch.nn.functional.normalize(feat)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": SD_REV,
        "adapter": IP_REV,
        "steps": 12,
        "size": 384,
        "seed": 13,
        "cases": [],
        "status": "in_progress",
    }
    selected = [case for case in items() if case["id"] != "B083V9C8D2"]
    manifest["expected_cases"] = 1 if demo else 2 * len(selected)
    manifest["protocol"] = "sd15-four-channel-rgb-gallery-v1"
    manifest["device"] = torch.cuda.get_device_name()
    manifest["environment"] = {"python":platform.python_version(),"cuda":torch.version.cuda,
                               **{key:version(key) for key in ("torch","diffusers","transformers","accelerate")}}
    previous = out/"manifest.json"
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old.get("status")=="in_progress" and all(
                old.get(k)==manifest[k] for k in ("model","adapter","steps","size","seed","expected_cases","protocol","device")):
            required = ("source.png","reference.png","reference-mask.png","mask.png","text.png",
                        "reference-edited.png","regional.png","identity_embeddings.npz")
            current = {item["id"]:item for item in selected}
            for case in old["cases"]:
                item=current.get(case["id"])
                if item is None or not all((out/case["case"]/p).is_file() for p in required):
                    continue
                if case.get("source_sha256")!=item["photo_sha256"] or case.get("reference_sha256")!=item["reference_sha256"]:
                    continue
                manifest["cases"].append(case)
    completed = {c["case"] for c in manifest["cases"]}
    write_manifest(out,manifest)
    for item in selected[:1] if demo else selected:
        source, mask, _ = prepare(item)
        reference, ref_mask, _ = prepare(item, ref=True)
        empty = Image.new("RGB", source.size, "white")
        ref_embedding = encode(Image.composite(reference,empty,ref_mask))
        for color in ["red"] if demo else ["red", "blue"]:
            key = item["id"] + "-" + color
            if key in completed:
                continue
            sub = out / key
            sub.mkdir(exist_ok=True)
            source.save(sub / "source.png")
            reference.save(sub / "reference.png")
            ref_mask.save(sub / "reference-mask.png")
            mask.save(sub / "mask.png")
            embeddings = {"reference_subject": ref_embedding[0].numpy()}
            timing = {}
            for name, scale in (("text", 0.0), ("reference", 0.6), ("regional", 0.6)):
                pipe.set_ip_adapter_scale(scale)
                args = {}
                if name == "regional":
                    processed = IPAdapterMaskProcessor().preprocess(
                        mask, height=384, width=384
                    )
                    args["cross_attention_kwargs"] = {
                        "ip_adapter_masks": [processed.reshape(1, 1, 384, 384)]
                    }
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = perf_counter()
                image = pipe(
                    prompt=f"A photo of the same {item['name']} on a plain {color} studio background",
                    image=source,
                    mask_image=ImageOps.invert(mask),
                    ip_adapter_image=reference,
                    height=384,
                    width=384,
                    num_inference_steps=12,
                    generator=torch.Generator("cuda").manual_seed(13),
                    **args,
                ).images[0]
                torch.cuda.synchronize()
                timing[name] = {"seconds":perf_counter()-started,
                                "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated()}
                image.save(
                    sub
                    / (("reference-edited" if name == "reference" else name) + ".png")
                )
                embeddings[name] = encode(Image.composite(image, empty, mask))[
                    0
                ].numpy()
            np.savez_compressed(sub / "identity_embeddings.npz", **embeddings)
            manifest["cases"].append(
                {"case": key, "id": item["id"], "split": item["split"], "color": color,
                 "source_sha256":item["photo_sha256"],"reference_sha256":item["reference_sha256"],
                 "timing":timing}
            )
            write_manifest(out,manifest)
    manifest["status"] = "completed"
    manifest["cases"].sort(key=lambda c:(c["split"]!="dev",c["id"],c["color"]))
    write_manifest(out,manifest)
    evaluate(out)
