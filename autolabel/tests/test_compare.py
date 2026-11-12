import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autolabel" / "nuclio"))
sys.path.insert(0, str(ROOT / "autolabel"))
try:
    import pycocotools  # noqa: F401
    import compare
    import core
except ImportError:
    compare = None

H, W = 40, 60


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def write_raw(here, task, stem, objs):
    d = here / "raw" / task
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps({"task": task, "image": stem, "objects": [
        {"label": l, "score": 0.9, "bbox": core.bbox(m), "rle": core.rle_encode(m)} for l, m in objs]}))


def write_coco_zip(path, stem, objs, classes):
    cats = {c: i + 1 for i, c in enumerate(classes)}
    coco = {"images": [{"id": 1, "file_name": f"{stem}.jpg", "height": H, "width": W}],
            "categories": [{"id": i, "name": c} for c, i in cats.items()],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": cats[l], "iscrowd": 0,
                             "segmentation": core.rle_encode(m)} for i, (l, m) in enumerate(objs)]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("annotations/instances_default.json", json.dumps(coco))


@unittest.skipIf(compare is None, "pycocotools not installed (pip install -r requirements.txt)")
class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.old = compare.HERE, compare.SUBMISSIONS
        compare.HERE, compare.SUBMISSIONS = self.tmp / "autolabel", self.tmp / "submissions"
        compare.SUBMISSIONS.mkdir(parents=True)

    def tearDown(self):
        compare.HERE, compare.SUBMISSIONS = self.old
        shutil.rmtree(self.tmp)

    def run_compare(self, argv=()):
        with redirect_stdout(io.StringIO()):
            return {r["task"]: r for r in compare.main(list(argv))}

    def test_instance_counts(self):
        cls = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        a, b, c = box(0, 10, 0, 10), box(0, 10, 20, 30), box(20, 30, 0, 10)
        write_raw(compare.HERE, "medium_instance", "img", [("car", a), ("car", b), ("car", c)])
        # kept a as is, b slightly edited, deleted c, drew a new person
        write_coco_zip(compare.SUBMISSIONS / "medium_instance.zip", "img",
                       [("car", a), ("car", box(0, 10, 21, 30)), ("person", box(30, 40, 30, 40))], cls)
        r = self.run_compare()["medium_instance"]
        self.assertEqual((r["final_objects"], r["from_tool"], r["added"], r["deleted"]), (3, 2, 1, 1))
        self.assertAlmostEqual(r["pct"], 2 / 3)
        self.assertTrue((compare.HERE / "AUTOLABEL_REPORT.md").is_file())

    def test_class_change_is_not_from_tool(self):
        cls = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        write_raw(compare.HERE, "cp1_holes", "img", [("car", box(0, 10, 0, 10))])
        write_coco_zip(compare.SUBMISSIONS / "cp1_holes.zip", "img", [("truck", box(0, 10, 0, 10))], cls)
        self.assertEqual(self.run_compare()["cp1_holes"]["from_tool"], 0)

    def test_semantic_pixels(self):
        write_raw(compare.HERE, "cp4_curb", "img", [("road", box(0, 20, 0, W)), ("sidewalk", box(20, 40, 0, W))])
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[:20] = (128, 64, 128)       # road, kept
        rgb[20:30] = (128, 64, 128)     # tool said sidewalk, user changed to road
        rgb[30:] = (244, 35, 232)       # sidewalk, kept
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        with zipfile.ZipFile(compare.SUBMISSIONS / "cp4_curb.zip", "w") as z:
            z.writestr("labelmap.txt", "# label:color_rgb:parts:actions\nbackground:0,0,0::\n"
                                       "road:128,64,128::\nsidewalk:244,35,232::\n")
            z.writestr("SegmentationClass/img.png", buf.getvalue())
        r = self.run_compare()["cp4_curb"]
        self.assertAlmostEqual(r["px"], 30 / 40)

    def test_panoptic_things_and_stuff(self):
        cls = json.loads((ROOT / "data/tiers/hard_panoptic/classes.json").read_text())["classes"]
        road, car = box(20, 40, 0, W), box(0, 10, 0, 10)
        write_raw(compare.HERE, "hard_panoptic", "img", [("road", road), ("car", car)])
        write_coco_zip(compare.SUBMISSIONS / "hard_panoptic.zip", "img",
                       [("road", road), ("sky", box(0, 20, 0, W)), ("car", car)], cls)
        r = self.run_compare()["hard_panoptic"]
        self.assertEqual((r["final_objects"], r["from_tool"]), (1, 1))
        # car overlaps the user's sky; stuff pixels: road 20*60 kept, sky 20*60 manual
        self.assertAlmostEqual(r["px"], 0.5)
        self.assertAlmostEqual(r["pct"], 0.75)

    def test_gt_scores_raw_masks(self):
        cls = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        a = box(0, 10, 0, 10)
        write_raw(compare.HERE, "medium_instance", "img", [("car", a)])
        write_coco_zip(compare.SUBMISSIONS / "medium_instance.zip", "img", [("car", a)], cls)
        gt_dir = self.tmp / "ref" / "data/tiers/medium_instance/groundtruth"
        gt_dir.mkdir(parents=True)
        with zipfile.ZipFile(compare.SUBMISSIONS / "medium_instance.zip") as z:
            (gt_dir / "instances.json").write_bytes(z.read("annotations/instances_default.json"))
        r = self.run_compare(["--gt", str(self.tmp / "ref")])["medium_instance"]
        self.assertAlmostEqual(r["gt"], 1.0)


if __name__ == "__main__":
    unittest.main()
