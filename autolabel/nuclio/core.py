"""Pure numpy/PIL logic of the SAM 3 auto-label function (no torch, so it is testable on the host)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

THUMB = 32


def thumbnail(img):
    return np.asarray(img.convert("L").resize((THUMB, THUMB), Image.BILINEAR), np.float32)


def build_index(data_dir):
    """[(task, stem, thumbnail)] for every lab image under data/tiers and data/checkpoints."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())["tasks"]
    index = []
    for task, info in manifest.items():
        for f in sorted((data_dir / info["path"] / "images").glob("*.jpg")):
            index.append((task, f.stem, thumbnail(Image.open(f))))
    return index


def match_image(img, index, max_dist):
    """(task, stem) of the closest lab image, or (None, None) if mean abs diff > max_dist."""
    t = thumbnail(img)
    dist, task, stem = min((float(np.abs(t - th).mean()), task, stem) for task, stem, th in index)
    return (task, stem) if dist <= max_dist else (None, None)


def task_classes(data_dir, task):
    """(type, classes, stuff) from the task's classes.json."""
    data_dir = Path(data_dir)
    path = json.loads((data_dir / "manifest.json").read_text())["tasks"][task]["path"]
    meta = json.loads((data_dir / path / "classes.json").read_text())
    return meta["type"], meta["classes"], meta.get("stuff", [])


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def postprocess_instance(dets, min_area, nms_iou, cross_class=False):
    """dets: [(label, score, bool HxW)]. Drop tiny masks, mask-IoU NMS within each class (across classes if cross_class)."""
    kept = []
    for label, score, m in sorted(dets, key=lambda d: -d[1]):
        if m.sum() < min_area:
            continue
        if any((cross_class or k[0] == label) and mask_iou(k[2], m) > nms_iou for k in kept):
            continue
        kept.append((label, score, m))
    return kept


def postprocess_semantic(dets, shape, min_area):
    """Each pixel goes to the class with the highest score covering it; one mask per class."""
    labels = sorted({d[0] for d in dets})
    if not labels:
        return []
    best = np.zeros((len(labels),) + tuple(shape), np.float32)
    for label, score, m in dets:
        i = labels.index(label)
        best[i][m] = np.maximum(best[i][m], score)
    winner = best.argmax(0)
    covered = best.max(0) > 0
    out = []
    for i, label in enumerate(labels):
        m = covered & (winner == i)
        if m.sum() >= min_area:
            out.append((label, float(best[i][m].max()), m))
    return out


def postprocess_panoptic(dets, shape, stuff, min_area, nms_iou):
    """Stuff as semantic, things as instance; things are cut out of stuff (thing wins).

    Things use cross-class NMS: a panoptic pixel has one owner, so a car also detected as truck is a duplicate.
    """
    things = postprocess_instance([d for d in dets if d[0] not in stuff], min_area, nms_iou, cross_class=True)
    stuff_out = postprocess_semantic([d for d in dets if d[0] in stuff], shape, min_area)
    if things:
        thing_px = np.logical_or.reduce([t[2] for t in things])
        stuff_out = [(l, s, m & ~thing_px) for l, s, m in stuff_out]
        stuff_out = [d for d in stuff_out if d[2].sum() >= min_area]
    return stuff_out + things


def bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def to_cvat_mask(m):
    """CVAT detector 'mask' field: bbox crop flattened row-major + [xtl, ytl, xbr, ybr] (inclusive).

    Same layout as serverless/openvino/base/shared.py::to_cvat_mask in CVAT v2.74.1; CVAT itself
    turns it into RLE in lambda_manager/views.py::_parse_anno.
    """
    xtl, ytl, xbr, ybr = bbox(m)
    return m[ytl:ybr + 1, xtl:xbr + 1].astype(np.uint8).ravel().tolist() + [xtl, ytl, xbr, ybr]


def rle_encode(m):
    """COCO uncompressed RLE (column-major, starts with a run of 0s); pycocotools can read it."""
    flat = np.asarray(m, bool).ravel(order="F")
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    edges = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(edges).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {"size": list(m.shape), "counts": counts}


def rle_decode(rle):
    h, w = rle["size"]
    vals = np.arange(len(rle["counts"])) % 2 == 1
    return np.repeat(vals, rle["counts"]).reshape((h, w), order="F")
