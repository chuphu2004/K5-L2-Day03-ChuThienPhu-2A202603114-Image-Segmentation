"""How much of each submitted task comes from the SAM 3 tool.

Compares the raw tool masks in autolabel/raw/<task>/ with the corrected CVAT exports in
submissions/<task>.zip and writes autolabel/AUTOLABEL_REPORT.md.

  objects (instance, panoptic things): a final object counts as "from the tool" if a tool mask of
      the same class overlaps it with IoU >= 0.5 (one-to-one, greedy).
  pixels (semantic, panoptic stuff): share of labeled final pixels that keep the class the tool gave.
  panoptic task % = mean of the two. Total = mean over tasks weighted by task points.

Usage:  python3 autolabel/compare.py [--gt DIR]
  --gt DIR  root of the unpacked official reference ZIP (DIR/data/tiers/<task>/groundtruth/...),
            kept OUTSIDE this repo. Also scores the raw tool masks with lab_utils metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lab_utils as L  # noqa: E402

DATA = ROOT / "data"
HERE = ROOT / "autolabel"
SUBMISSIONS = ROOT / "submissions"


def raw_coco(raw_dir, class_names):
    """autolabel/raw/<task>/*.json -> a COCO dict lab_utils can read."""
    cats = {c: i + 1 for i, c in enumerate(class_names)}
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for c, i in cats.items()]}
    for img_id, f in enumerate(sorted(Path(raw_dir).glob("*.json")), 1):
        raw = json.loads(f.read_text())
        objs = [o for o in raw["objects"] if o["label"] in cats]
        h, w = objs[0]["rle"]["size"] if objs else (1, 1)
        coco["images"].append({"id": img_id, "file_name": raw["image"] + ".jpg", "height": h, "width": w})
        for o in objs:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id,
                                        "category_id": cats[o["label"]], "segmentation": o["rle"],
                                        "score": o["score"]})
    return coco


def count_kept(tool, final):
    """Final objects matched one-to-one to a same-class tool mask with IoU >= 0.5."""
    used, kept = set(), 0
    for fc, fm, _ in final:
        best, bj = 0.5, -1
        for j, (tc, tm, _) in enumerate(tool):
            if j not in used and tc == fc:
                v = L._iou(fm, tm)
                if v >= best:
                    best, bj = v, j
        if bj >= 0:
            used.add(bj)
            kept += 1
    return kept


def paint(items, name2id, shape):
    lab = np.full(shape, 255, np.int32)
    for cls, m, _ in sorted(items, key=lambda t: -int(t[1].sum())):  # small last, like lab_utils
        lab[m] = name2id[cls]
    return lab


def compare_task(name, info, sub_path):
    meta = json.loads((DATA / info["path"] / "classes.json").read_text())
    classes, stuff, ttype = meta["classes"], set(meta.get("stuff", [])), meta["type"]
    tool = L._instances_by_image(raw_coco(HERE / "raw" / name, classes), set(classes))
    row = {"task": name, "type": ttype, "weight": info["weight"]}
    if ttype == "semantic":
        name2id = meta["trainid"]
        final = L.load_semantic_submission(sub_path, name2id)
        kept = labeled = 0
        for stem, f in final.items():
            t = paint(tool.get(stem, []), name2id, f.shape)
            labeled += int((f != 255).sum())
            kept += int(((f != 255) & (f == t)).sum())
        row["px"] = kept / labeled if labeled else 0.0
        row["pct"] = row["px"]
        return row
    coco = L._find_coco_json(L._read_members(sub_path))
    final = L._instances_by_image(coco, set(classes))
    n_final = n_tool = kept = px_kept = px_labeled = 0
    stuff_ids = {c: i for i, c in enumerate(sorted(stuff))}
    dims = {Path(im["file_name"]).stem: (im["height"], im["width"]) for im in coco["images"]}
    for stem in set(final) | set(tool):
        f_items, t_items = final.get(stem, []), tool.get(stem, [])
        f_things = [x for x in f_items if x[0] not in stuff]
        t_things = [x for x in t_items if x[0] not in stuff]
        if stem in final:
            n_final += len(f_things)
            kept += count_kept(t_things, f_things)
        n_tool += len(t_things)
        if stuff and stem in dims:
            f = paint([x for x in f_items if x[0] in stuff], stuff_ids, dims[stem])
            t = paint([x for x in t_items if x[0] in stuff], stuff_ids, dims[stem])
            px_labeled += int((f != 255).sum())
            px_kept += int(((f != 255) & (f == t)).sum())
    row.update(final_objects=n_final, from_tool=kept, added=n_final - kept, deleted=n_tool - kept)
    row["obj"] = kept / n_final if n_final else 0.0
    if stuff:
        row["px"] = px_kept / px_labeled if px_labeled else 0.0
        row["pct"] = (row["obj"] + row["px"]) / 2
    else:
        row["pct"] = row["obj"]
    return row


def score_raw(name, info, gt_root):
    """lab_utils metric of the raw tool masks against the reference (never stored in the repo)."""
    meta = json.loads((DATA / info["path"] / "classes.json").read_text())
    classes, ttype = meta["classes"], meta["type"]
    gt_dir = Path(gt_root) / "data" / info["path"] / "groundtruth"
    coco = raw_coco(HERE / "raw" / name, classes)
    if ttype == "semantic":
        name2id = meta["trainid"]
        tool = L._instances_by_image(coco, set(classes))
        dims = {Path(im["file_name"]).stem: (im["height"], im["width"]) for im in coco["images"]}
        sub = {s: paint(tool.get(s, []), name2id, hw) for s, hw in dims.items()}
        gt = L.load_semantic_gt(gt_dir, set(name2id.values()))
        return L.score_semantic(sub, gt, {v: k for k, v in name2id.items()})["value"]
    if ttype == "panoptic":
        cn, stuff = set(classes), set(meta["stuff"])
        catid2name = {int(k): v for k, v in meta["catid2name"].items()}
        gt = L.load_panoptic_gt(gt_dir / "panoptic.json", gt_dir / "png", catid2name, cn)
        return L.score_panoptic(L.build_pred_panoptic(coco, cn, stuff), gt, cn)["value"]
    gt = json.loads((gt_dir / "instances.json").read_text())
    return L.score_instance(coco, gt, set(classes))["value"]


def fmt(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="root of the unpacked reference ZIP, outside the repo")
    args = ap.parse_args(argv)
    tasks = json.loads((DATA / "manifest.json").read_text())["tasks"]
    rows = []
    for name, info in tasks.items():
        sub = next((p for p in (SUBMISSIONS / f"{name}.zip", SUBMISSIONS / name)
                    if p.exists()), None)
        if not (HERE / "raw" / name).is_dir():
            print(f"skip {name}: no raw tool output in autolabel/raw/{name}/")
            continue
        if sub is None:
            print(f"skip {name}: no submission {name}.zip")
            continue
        row = compare_task(name, info, sub)
        if args.gt and (Path(args.gt) / "data" / info["path"] / "groundtruth").is_dir():  # checkpoints have no GT
            row["gt"] = score_raw(name, info, args.gt)
        rows.append(row)
    if not rows:
        print("nothing to compare")
        return rows

    lines = ["# SAM 3 auto-label — phần do tool làm", "",
             "| Task | Loại | Điểm | Object cuối | Từ tool | Tự vẽ thêm | Tool bị xoá | % object | % pixel | % task |"
             + (" Mask gốc vs GT |" if args.gt else ""),
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" + (" ---: |" if args.gt else "")]
    for r in rows:
        lines.append(f"| {r['task']} | {r['type']} | {r['weight']} | {r.get('final_objects', '—')} "
                     f"| {r.get('from_tool', '—')} | {r.get('added', '—')} | {r.get('deleted', '—')} "
                     f"| {fmt(r.get('obj'))} | {fmt(r.get('px'))} | {fmt(r['pct'])} |"
                     + ((f" {r['gt']:.3f} |" if "gt" in r else " — |") if args.gt else ""))
    total = sum(r["pct"] * r["weight"] for r in rows) / sum(r["weight"] for r in rows)
    lines += ["", f"**Tổng (trung bình có trọng số theo điểm task, {len(rows)} task): {fmt(total)}**", ""]
    text = "\n".join(lines)
    print(text)
    (HERE / "AUTOLABEL_REPORT.md").write_text(text)
    print(f"wrote {HERE / 'AUTOLABEL_REPORT.md'}")
    return rows


if __name__ == "__main__":
    main()
