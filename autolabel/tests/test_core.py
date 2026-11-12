import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autolabel" / "nuclio"))
import core  # noqa: E402

MAX_DIST = 12  # autolabel/config.json match_max_dist


def box(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


class ThumbnailMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = core.build_index(ROOT / "data")

    def test_all_14_images_indexed(self):
        self.assertEqual(len(self.index), 14)

    def test_each_image_recompressed_by_cvat_matches_itself(self):
        for task, stem, _ in self.index:
            path = next((ROOT / "data").glob(f"*/{task}/images/{stem}.jpg"))
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, "JPEG", quality=50)  # CVAT compressed chunk
            self.assertEqual(core.match_image(Image.open(buf), self.index, MAX_DIST), (task, stem))

    def test_distinct_images_are_far_apart(self):
        d = min(float(np.abs(a[2] - b[2]).mean())
                for i, a in enumerate(self.index) for b in self.index[i + 1:])
        self.assertGreater(d, 2 * MAX_DIST)

    def test_unknown_image(self):
        noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), np.uint8))
        self.assertEqual(core.match_image(noise, self.index, MAX_DIST), (None, None))

    def test_task_classes(self):
        ttype, classes, stuff = core.task_classes(ROOT / "data", "hard_panoptic")
        self.assertEqual(ttype, "panoptic")
        self.assertIn("car", classes)
        self.assertIn("road", stuff)


class PostProcess(unittest.TestCase):
    S = (100, 100)

    def test_instance_nms_and_min_area(self):
        dets = [("car", 0.9, box(self.S, 0, 40, 0, 40)),
                ("car", 0.6, box(self.S, 0, 40, 0, 38)),     # duplicate from 2nd prompt
                ("truck", 0.7, box(self.S, 0, 40, 0, 40)),   # other class, kept
                ("car", 0.8, box(self.S, 60, 62, 60, 62))]   # 4 px, too small
        out = core.postprocess_instance(dets, min_area=64, nms_iou=0.5)
        self.assertEqual([(l, s) for l, s, _ in out], [("car", 0.9), ("truck", 0.7)])

    def test_semantic_highest_score_wins_one_mask_per_class(self):
        dets = [("road", 0.9, box(self.S, 50, 100, 0, 100)),
                ("sidewalk", 0.5, box(self.S, 40, 70, 0, 100)),
                ("vegetation", 0.6, box(self.S, 0, 10, 0, 10)),
                ("vegetation", 0.7, box(self.S, 0, 10, 90, 100))]  # "tree" prompt, separate blob
        out = {l: m for l, _, m in core.postprocess_semantic(dets, self.S, min_area=1)}
        self.assertEqual(set(out), {"road", "sidewalk", "vegetation"})
        self.assertTrue(out["road"][50:].all())
        self.assertEqual(out["sidewalk"].sum(), 10 * 100)
        self.assertEqual(out["vegetation"].sum(), 200)
        self.assertFalse(np.logical_and(out["road"], out["sidewalk"]).any())

    def test_panoptic_thing_over_stuff(self):
        dets = [("road", 0.9, box(self.S, 50, 100, 0, 100)),
                ("car", 0.8, box(self.S, 40, 60, 10, 30)),
                ("car", 0.7, box(self.S, 40, 60, 60, 80))]
        out = core.postprocess_panoptic(dets, self.S, {"road"}, min_area=1, nms_iou=0.5)
        labels = [l for l, _, _ in out]
        self.assertEqual(sorted(labels), ["car", "car", "road"])
        road = out[labels.index("road")][2]
        self.assertEqual(road.sum(), 50 * 100 - 2 * 10 * 20)

    def test_panoptic_things_cross_class_nms(self):
        dets = [("car", 0.8, box(self.S, 40, 60, 10, 30)),
                ("truck", 0.6, box(self.S, 40, 60, 10, 30))]  # same vehicle as another class: dropped
        out = core.postprocess_panoptic(dets, self.S, set(), min_area=1, nms_iou=0.5)
        self.assertEqual([(l, s) for l, s, _ in out], [("car", 0.8)])


class Encoding(unittest.TestCase):
    def setUp(self):
        self.m = np.zeros((7, 9), bool)
        self.m[2:5, 3:8] = True
        self.m[0, 0] = True
        self.m[6, 8] = True

    def test_cvat_mask_layout(self):
        flat = core.to_cvat_mask(self.m)
        xtl, ytl, xbr, ybr = flat[-4:]
        self.assertEqual((xtl, ytl, xbr, ybr), (0, 0, 8, 6))
        crop = np.array(flat[:-4]).reshape(ybr - ytl + 1, xbr - xtl + 1)
        self.assertTrue((crop.astype(bool) == self.m[ytl:ybr + 1, xtl:xbr + 1]).all())
        self.assertTrue(set(flat[:-4]) <= {0, 1})

    def test_cvat_mask_crop_is_inclusive(self):
        flat = core.to_cvat_mask(self.m[2:5, 3:8].copy())
        self.assertEqual(flat[-4:], [0, 0, 4, 2])
        self.assertEqual(len(flat) - 4, 3 * 5)

    def test_rle_roundtrip(self):
        for m in (self.m, ~self.m, np.zeros((3, 4), bool), np.ones((3, 4), bool)):
            self.assertTrue((core.rle_decode(core.rle_encode(m)) == m).all())

    def test_rle_readable_by_pycocotools(self):
        try:
            from pycocotools import mask as cocomask
        except ImportError:
            self.skipTest("pycocotools not installed")
        rle = core.rle_encode(self.m)
        dec = cocomask.decode(cocomask.frPyObjects(rle, *rle["size"])).astype(bool)
        self.assertTrue((dec == self.m).all())


if __name__ == "__main__":
    unittest.main()
