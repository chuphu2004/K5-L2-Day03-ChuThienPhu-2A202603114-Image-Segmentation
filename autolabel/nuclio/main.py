"""Nuclio handler: SAM 3 text-prompted auto-labeling for the Day03 CVAT tasks."""
import base64
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

import core

ROOT = Path("/autolabel")  # host autolabel/ mounted read-write (config, weights, raw)
DATA = Path("/data")       # host data/ mounted read-only


def init_context(context):
    context.user_data.index = core.build_index(DATA)
    context.user_data.processor = None  # loaded on first call so deploy works before weights exist
    context.logger.info(f"indexed {len(context.user_data.index)} lab images")


def load_processor(ckpt):
    if not ckpt.is_file():
        raise RuntimeError(f"SAM 3 weights not found at autolabel/{ckpt.relative_to(ROOT)}; "
                           "run autolabel/download_weights.sh on the host (see autolabel/README.md)")
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False, device=device)
    return Sam3Processor(model, device=device)


def detect(proc, img, classes, cfg):
    """[(label, score, bool mask)] for every prompt of every class."""
    import torch
    dets = []
    # sam3's fused ViT MLP (perflib/fused.py addmm_act) returns bfloat16, so on GPU it runs under bf16 autocast like
    # upstream; the CPU image patches those casts out and stays float32 (CPU bf16 is slow).
    with torch.autocast(device_type=proc.device, dtype=torch.bfloat16, enabled=proc.device == "cuda"):
        state = proc.set_image(img)
        for cls in classes:
            proc.confidence_threshold = cfg["thresholds"].get(cls, cfg["default_threshold"])
            for prompt in cfg["prompts"].get(cls, [cls]):
                out = proc.set_text_prompt(prompt=prompt, state=state)
                for m, s in zip(out["masks"][:, 0].bool().cpu().numpy(), out["scores"].float().tolist()):
                    dets.append((cls, s, m))
    return dets


def handler(context, event):
    t0 = time.time()
    cfg = json.loads((ROOT / "config.json").read_text())
    img = Image.open(io.BytesIO(base64.b64decode(event.body["image"]))).convert("RGB")
    task, stem = core.match_image(img, context.user_data.index, cfg["match_max_dist"])
    if task:
        ttype, classes, stuff = core.task_classes(DATA, task)
    else:
        ttype, classes, stuff = "instance", list(cfg["prompts"]), []
        task, stem = "unknown", time.strftime("%Y%m%d-%H%M%S")
    context.logger.info(f"image -> task={task} stem={stem} type={ttype}")

    try:
        if context.user_data.processor is None:
            context.user_data.processor = load_processor(ROOT / cfg["checkpoint"])
        dets = detect(context.user_data.processor, img, classes, cfg)
    except Exception as e:
        context.logger.error(f"SAM 3 failed: {e}")
        return context.Response(body=json.dumps({"error": str(e)}), headers={},
                                content_type="application/json", status_code=500)

    shape = (img.height, img.width)
    if ttype == "semantic":
        objs = core.postprocess_semantic(dets, shape, cfg["min_area"])
    elif ttype == "panoptic":
        objs = core.postprocess_panoptic(dets, shape, set(stuff), cfg["min_area"], cfg["nms_iou"])
    else:
        objs = core.postprocess_instance(dets, cfg["min_area"], cfg["nms_iou"])

    raw = ROOT / "raw" / task / f"{stem}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({
        "task": task, "image": stem, "type": ttype, "model": "facebook/sam3 " + cfg["checkpoint"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
        "objects": [{"label": l, "score": round(s, 4), "bbox": core.bbox(m), "rle": core.rle_encode(m)}
                    for l, s, m in objs],
    }))
    owner = ROOT.stat()  # container runs as root; hand files back to the host user
    for p in (raw.parent.parent, raw.parent, raw):
        os.chown(p, owner.st_uid, owner.st_gid)
    context.logger.info(f"{len(objs)} objects in {time.time() - t0:.1f}s -> {raw}")

    results = [{"confidence": str(round(s, 4)), "label": l, "type": "mask", "mask": core.to_cvat_mask(m)}
               for l, s, m in objs]
    return context.Response(body=json.dumps(results), headers={},
                            content_type="application/json", status_code=200)
