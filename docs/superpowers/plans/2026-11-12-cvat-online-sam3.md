# SAM 3 cho CVAT online (REST API + PAT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lệnh `autolabel\run.ps1 -Target local|online -Task <id>` chạy SAM 3 trên GPU máy học viên và ghi mask vào một task CVAT bất kỳ qua REST API, đăng nhập bằng Personal Access Token.

**Architecture:** `sam3_func.py` bọc SAM 3 (dùng chung `nuclio/core.py` với function nuclio) thành auto-annotation function của `cvat-sdk`; `cvat_run.py` đọc task, chọn class/mode, kiểm an toàn rồi gọi `cvat_sdk.auto_annotation.annotate_task`. Chạy trong image `cvat.sam3.sdk:<version>` = image nền `cvat.sam3.base.gpu` + `cvat-sdk`, khởi động bằng `run.ps1`.

**Tech Stack:** Python 3.12, torch 2.10 cu128, `sam3` @ `f92a5ef`, `cvat-sdk` 2.74.1, numpy 1.26, Docker Desktop (WSL2, `--gpus all`), PowerShell 5.1, unittest.

**Spec:** [`docs/superpowers/specs/2026-11-12-cvat-online-sam3-design.md`](../specs/2026-11-12-cvat-online-sam3-design.md)

## Global Constraints

- Windows 11 + Docker Desktop (WSL2). **Không cài tool lên host**; chỉ Docker và venv `~\.venvs\day03`. Lệnh Docker chạy từ **PowerShell** (Git Bash đổi đường dẫn `/autolabel` thành đường dẫn Windows).
- Python trên host: `$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" ...` (cp1252 mặc định làm hỏng file tiếng Việt).
- Mọi code mới nằm trong `autolabel/`; ngoài ra chỉ sửa `.gitignore` và tài liệu trong `docs/superpowers/`. **Không sửa** `data/`, `tests/` ở gốc repo, `notebooks/`, `.github/`, `submissions/`, `scoring/`.
- `cvat-sdk` mặc định `2.74.1` (bằng CVAT local). Cài **không kèm extra** để giữ numpy 1.26 của image nền.
- PAT không bao giờ nằm trong git, image, log, raw hay chat. `autolabel/.env*` bị git-ignore trừ `autolabel/.env.example`.
- Raw của task không phải Day03 ghi vào `autolabel/raw/_online/` (git-ignore).
- Không bao giờ `docker compose down -v`.
- Thông báo do code in ra bằng **tiếng Anh** (console Windows); README tiếng Việt.
- Commit message không có dòng `Co-Authored-By` (quy ước của học viên). Không push nếu học viên chưa bảo.
- Tool không bao giờ tự xoá annotation nếu không có `--clear`.

---

## Task 0: Điều kiện bắt đầu và commit nền

**Files:** không sửa file.

- [ ] **Step 1: Kiểm Day03 local đã xong**

Chạy (PowerShell):
```powershell
docker images cvat.sam3.base.gpu --format "{{.Repository}}:{{.Tag}} {{.Size}}"
docker ps --format "{{.Names}}" | Select-String "cvat_server|nuclio-nuclio-sam3-autolabel"
```
Expected: có dòng `cvat.sam3.base.gpu:latest ...`; có `cvat_server` và `nuclio-nuclio-sam3-autolabel`. Thiếu thì **dừng**: Day03 local (subagent trước) chưa xong.

- [ ] **Step 2: Test hiện có đều pass**

```powershell
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -m unittest discover -s autolabel/tests
```
Expected: `Ran 17 tests` ... `OK`.

- [ ] **Step 3: Commit nền (`autolabel/`, `docs/superpowers/`, `.gitignore` chưa từng commit)**

Chỉ làm khi học viên đã duyệt plan này (duyệt plan = cho phép các commit local trong plan; không push).
```powershell
git status --short
git add .gitignore autolabel docs/superpowers
git status --short   # KHÔNG được có autolabel/weights/ hay file .env
git commit -m "feat(autolabel): SAM 3 nuclio auto-label for Day03 local CVAT"
```

---

## Task 1: Dùng chung phần SAM 3 trong `nuclio/core.py`

Chuyển `load_processor`, `detect` từ `nuclio/main.py` sang `core.py`, thêm `postprocess` (chọn hậu xử lý theo mode) và `raw_record` (nội dung file raw), để bản nuclio và bản SDK dùng chung.

**Files:**
- Modify: `autolabel/nuclio/core.py` (thêm `import time`, sửa docstring đầu file, thêm 4 hàm ở cuối)
- Modify: `autolabel/nuclio/main.py` (bỏ 2 hàm, handler gọi `core.*`)
- Modify: `autolabel/README.md` (1 dòng thông báo lỗi weights)
- Test: `autolabel/tests/test_core_sam3.py` (mới, chạy trên host)

**Interfaces:**
- Produces (dùng ở Task 3, 4):
  - `core.load_processor(ckpt: Path | str) -> Sam3Processor` — raise `RuntimeError("SAM 3 weights not found at <ckpt>; ...")` nếu thiếu file.
  - `core.detect(proc, img: PIL.Image, classes: list[str], cfg: dict) -> list[tuple[str, float, np.ndarray[bool, HxW]]]`
  - `core.postprocess(mode: str, dets, shape: tuple[int, int], stuff: list[str], cfg: dict) -> list[tuple[str, float, np.ndarray]]` với `mode` ∈ `semantic|instance|panoptic` (giá trị khác coi như `instance`).
  - `core.raw_record(task: str, stem: str, mode: str, cfg: dict, seconds: float, objs) -> dict` (format raw cũ, `compare.py` đọc được).
  - Test helpers trong `autolabel/tests/test_core_sam3.py`: `CFG: dict`, `box(y0, y1, x0, x1, shape=(10, 10)) -> np.ndarray[bool]`, `FakeProcessor(table: dict[str, list[tuple[float, np.ndarray]]])` (thuộc tính `calls: list[tuple[prompt, threshold]]`).

- [ ] **Step 1: Viết test (fail)**

Tạo `autolabel/tests/test_core_sam3.py`:
```python
"""SAM 3 helpers shared by the nuclio handler and the cvat-sdk function (no torch needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "autolabel" / "nuclio"))
import core  # noqa: E402

CFG = {"checkpoint": "weights/sam3.pt", "default_threshold": 0.4, "thresholds": {"car": 0.6},
       "min_area": 4, "nms_iou": 0.5, "prompts": {"vegetation": ["vegetation", "tree"]}}


def box(y0, y1, x0, x1, shape=(10, 10)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


class FakeTensor:
    """Mimics the torch tensors Sam3Processor returns: indexing, .cpu(), .numpy(), .tolist()."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, k):
        return FakeTensor(self.a[k])

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def tolist(self):
        return self.a.tolist()


class FakeProcessor:
    """prompt -> [(score, 10x10 bool mask)]; records (prompt, threshold) for every text prompt."""

    def __init__(self, table):
        self.table, self.calls, self.confidence_threshold = table, [], None

    def set_image(self, img):
        return {"image": img}

    def set_text_prompt(self, prompt, state):
        self.calls.append((prompt, self.confidence_threshold))
        hits = self.table.get(prompt, [])
        masks = np.array([m for _, m in hits], bool).reshape(len(hits), 1, 10, 10)
        return {"masks": FakeTensor(masks), "scores": FakeTensor([s for s, _ in hits])}


class Detect(unittest.TestCase):
    def test_prompts_thresholds_and_labels(self):
        proc = FakeProcessor({"car": [(0.9, box(0, 4, 0, 4))], "tree": [(0.7, box(5, 9, 5, 9))]})
        dets = core.detect(proc, "img", ["car", "vegetation"], CFG)
        self.assertEqual(proc.calls, [("car", 0.6), ("vegetation", 0.4), ("tree", 0.4)])
        self.assertEqual([(l, s) for l, s, _ in dets], [("car", 0.9), ("vegetation", 0.7)])
        self.assertTrue(dets[1][2][6, 6])


class Postprocess(unittest.TestCase):
    def test_dispatch_by_mode(self):
        dets = [("road", 0.8, box(0, 10, 0, 10)), ("car", 0.9, box(2, 5, 2, 5)),
                ("car", 0.85, box(2, 5, 2, 5))]
        self.assertEqual(len(core.postprocess("instance", dets, (10, 10), [], CFG)), 2)  # road + 1 car
        sem = core.postprocess("semantic", dets, (10, 10), [], CFG)
        self.assertEqual(sorted(l for l, _, _ in sem), ["car", "road"])
        pan = core.postprocess("panoptic", dets, (10, 10), ["road"], CFG)
        road = next(m for l, _, m in pan if l == "road")
        self.assertFalse(road[3, 3])  # thing cut out of stuff


class RawRecord(unittest.TestCase):
    def test_fields_and_rle_roundtrip(self):
        m = box(1, 3, 2, 6)
        rec = core.raw_record("cp2_slice", "img1", "instance", CFG, 1.234, [("car", 0.91234, m)])
        self.assertEqual((rec["task"], rec["image"], rec["type"], rec["seconds"]),
                         ("cp2_slice", "img1", "instance", 1.2))
        obj = rec["objects"][0]
        self.assertEqual((obj["label"], obj["score"], obj["bbox"]), ("car", 0.9123, [2, 1, 5, 2]))
        self.assertTrue((core.rle_decode(obj["rle"]) == m).all())
```

- [ ] **Step 2: Chạy test, phải fail**

```powershell
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -m unittest discover -s autolabel/tests -p "test_core_sam3.py"
```
Expected: FAIL/ERROR `AttributeError: module 'core' has no attribute 'detect'`.

- [ ] **Step 3: Thêm hàm vào `core.py`**

Thay dòng docstring đầu `autolabel/nuclio/core.py`:
```python
"""Pure numpy/PIL logic of the SAM 3 auto-label function (no torch, so it is testable on the host)."""
import json
```
bằng:
```python
"""Numpy/PIL logic of the SAM 3 auto-label function; torch/sam3 are imported only inside load_processor."""
import json
import time
```
Thêm vào **cuối** file:
```python


def load_processor(ckpt):
    """SAM 3 image model + processor, on cuda when available."""
    ckpt = Path(ckpt)
    if not ckpt.is_file():
        raise RuntimeError(f"SAM 3 weights not found at {ckpt}; "
                           "run autolabel/download_weights.sh on the host (see autolabel/README.md)")
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False, device=device)
    return Sam3Processor(model, device=device)


def detect(proc, img, classes, cfg):
    """[(label, score, bool mask)] for every prompt of every class."""
    state = proc.set_image(img)
    dets = []
    for cls in classes:
        proc.confidence_threshold = cfg["thresholds"].get(cls, cfg["default_threshold"])
        for prompt in cfg["prompts"].get(cls, [cls]):
            out = proc.set_text_prompt(prompt=prompt, state=state)
            for m, s in zip(out["masks"][:, 0].cpu().numpy(), out["scores"].tolist()):
                dets.append((cls, s, m))
    return dets


def postprocess(mode, dets, shape, stuff, cfg):
    if mode == "semantic":
        return postprocess_semantic(dets, shape, cfg["min_area"])
    if mode == "panoptic":
        return postprocess_panoptic(dets, shape, set(stuff), cfg["min_area"], cfg["nms_iou"])
    return postprocess_instance(dets, cfg["min_area"], cfg["nms_iou"])


def raw_record(task, stem, mode, cfg, seconds, objs):
    """Content of autolabel/raw/<task>/<stem>.json (read by compare.py)."""
    return {
        "task": task, "image": stem, "type": mode, "model": "facebook/sam3 " + cfg["checkpoint"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(seconds, 1),
        "objects": [{"label": l, "score": round(s, 4), "bbox": bbox(m), "rle": rle_encode(m)}
                    for l, s, m in objs],
    }
```

- [ ] **Step 4: Chạy test, phải pass**

```powershell
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -m unittest discover -s autolabel/tests
```
Expected: `Ran 20 tests` ... `OK`.

- [ ] **Step 5: `main.py` dùng hàm chung**

Đọc `autolabel/nuclio/main.py` hiện tại. Nếu khác bản dưới ở chỗ không liên quan (fix do subagent Day03 thêm sau), giữ fix đó. Nội dung đích của `autolabel/nuclio/main.py`:
```python
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
            context.user_data.processor = core.load_processor(ROOT / cfg["checkpoint"])
        dets = core.detect(context.user_data.processor, img, classes, cfg)
    except Exception as e:
        context.logger.error(f"SAM 3 failed: {e}")
        return context.Response(body=json.dumps({"error": str(e)}), headers={},
                                content_type="application/json", status_code=500)

    objs = core.postprocess(ttype, dets, (img.height, img.width), stuff, cfg)

    raw = ROOT / "raw" / task / f"{stem}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(core.raw_record(task, stem, ttype, cfg, time.time() - t0, objs)))
    owner = ROOT.stat()  # container runs as root; hand files back to the host user
    for p in (raw.parent.parent, raw.parent, raw):
        os.chown(p, owner.st_uid, owner.st_gid)
    context.logger.info(f"{len(objs)} objects in {time.time() - t0:.1f}s -> {raw}")

    results = [{"confidence": str(round(s, 4)), "label": l, "type": "mask", "mask": core.to_cvat_mask(m)}
               for l, s, m in objs]
    return context.Response(body=json.dumps(results), headers={},
                            content_type="application/json", status_code=200)
```
Trong `autolabel/README.md`, đổi `SAM 3 weights not found at autolabel/weights/sam3.pt` thành `SAM 3 weights not found at /autolabel/weights/sam3.pt`.

- [ ] **Step 6: Deploy lại nuclio và smoke test (hành vi không đổi)**

Trước khi deploy, giữ raw cũ để so:
```powershell
Copy-Item autolabel\raw\cp2_slice\000000017627.json "$env:TEMP\before_cp2.json"
powershell -File autolabel\deploy.ps1 gpu
$port = ((docker port nuclio-nuclio-sam3-autolabel 8080) -split "`n")[0].Split(":")[-1]
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -c "import base64,json,sys,urllib.request as u;b=base64.b64encode(open('data/checkpoints/cp2_slice/images/000000017627.jpg','rb').read()).decode();r=u.urlopen(u.Request('http://localhost:'+sys.argv[1],json.dumps({'image':b}).encode(),{'Content-Type':'application/json'}),timeout=600);print(len(json.load(r)),'objects')" $port
& "$HOME\.venvs\day03\Scripts\python.exe" -c "import json,sys,collections as c;f=lambda p:c.Counter(o['label'] for o in json.load(open(p))['objects']);a,b=f(sys.argv[1]),f(sys.argv[2]);print(a,b);sys.exit(a!=b)" "$env:TEMP\before_cp2.json" autolabel\raw\cp2_slice\000000017627.json
```
Expected: in số object; hai `Counter` giống nhau, exit 0. `docker logs --tail 5 nuclio-nuclio-sam3-autolabel` có `objects in`.

- [ ] **Step 7: Commit**

```powershell
git add autolabel/nuclio/core.py autolabel/nuclio/main.py autolabel/README.md autolabel/tests/test_core_sam3.py
git commit -m "refactor(autolabel): share SAM 3 detect/postprocess/raw helpers in core"
```

---

## Task 2: Image `cvat.sam3.sdk` và `run.ps1`

**Files:**
- Create: `autolabel/Dockerfile.sdk`
- Create: `autolabel/run.ps1`

**Interfaces:**
- Consumes: image `cvat.sam3.base.gpu` (Task 0).
- Produces: image `cvat.sam3.sdk:<SdkVersion>`; `powershell -File autolabel\run.ps1 [-Target local|online] -Task <id> [-Check] [-Mode m] [-Stuff a,b] [-Append|-Clear] [-SdkVersion 2.74.1]` (đổi thành cờ `--task/--check/--mode/--stuff/--append/--clear` của `cvat_run.py`, Task 4) và `run.ps1 -Test`. Trong container: `autolabel/` → `/autolabel` (rw), `data/` → `/data` (ro), làm việc ở `/autolabel`.

- [ ] **Step 1: Kiểm `cvat-sdk` 2.74.1 có trên PyPI**

```powershell
docker run --rm cvat.sam3.base.gpu pip index versions cvat-sdk
```
Expected: dòng `Available versions:` chứa `2.74.1`. Không có → chọn bản `2.74.x` cao nhất có trong danh sách và dùng nó làm mặc định `ARG`/`-SdkVersion` ở các bước sau.

- [ ] **Step 2: Tạo `autolabel/Dockerfile.sdk`**

```dockerfile
# SAM 3 + cvat-sdk for cvat_run.py (CVAT REST API + Personal Access Token). Built by run.ps1, no build context.
# cvat-sdk major.minor should match the CVAT server (local CVAT is v2.74.1). No [masks] extra: keep sam3's numpy 1.x.
FROM cvat.sam3.base.gpu
ARG CVAT_SDK_VERSION=2.74.1
RUN pip install --no-cache-dir "cvat-sdk==${CVAT_SDK_VERSION}" "numpy>=1.26,<2"
WORKDIR /autolabel
```

- [ ] **Step 3: Tạo `autolabel/run.ps1`**

```powershell
# Run SAM 3 on a CVAT task through the REST API (cvat_run.py) inside the GPU container.
# Usage:
#   powershell -File autolabel\run.ps1 [-Target local|online] -Task <id> [-Check] [-Mode semantic|instance|panoptic] [-Stuff road,sky] [-Append|-Clear] [-SdkVersion 2.74.1]
#   powershell -File autolabel\run.ps1 -Test      # unit tests that need cvat-sdk (autolabel/tests/test_sdk*.py)
# Credentials come only from autolabel\.env.<Target> (git-ignored), passed with --env-file.
param(
    [string]$Target = "local",
    [int]$Task,
    [switch]$Check,
    [string]$Mode = "",
    [string[]]$Stuff = @(),
    [switch]$Append,
    [switch]$Clear,
    [string]$SdkVersion = "2.74.1",
    [switch]$Test
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Image = "cvat.sam3.sdk:$SdkVersion"

if (-not (docker images -q $Image)) {
    # Dockerfile from stdin = no build context (autolabel/ holds 3.4 GB of weights).
    Get-Content -Raw "$Here\Dockerfile.sdk" | docker build -t $Image --build-arg "CVAT_SDK_VERSION=$SdkVersion" -
    if ($LASTEXITCODE) { throw "docker build failed" }
}
$DockerArgs = @("run", "--rm", "--gpus", "all",
    "--mount", "type=bind,source=$Here,target=/autolabel",
    "--mount", "type=bind,source=$Repo\data,target=/data,readonly")
# Local CVAT: join its network so CVAT_URL=http://cvat_server:8080 resolves. Harmless for an online server.
if (docker network ls -q --filter "name=^cvat_cvat$") { $DockerArgs += @("--network", "cvat_cvat") }

if ($Test) {
    docker @DockerArgs $Image python -m unittest discover -s /autolabel/tests -p "test_sdk*.py" -v
} else {
    if (-not $Task) { throw "-Task <id> is required (the number in .../tasks/<id>)" }
    $EnvFile = "$Here\.env.$Target"
    if (-not (Test-Path $EnvFile)) { throw "Missing $EnvFile - copy autolabel\.env.example to it and fill it in" }
    # PowerShell switches -> cvat_run.py flags (argparse validates values, -Append/-Clear are exclusive there).
    $RunArgs = @("--task", "$Task")
    if ($Check) { $RunArgs += "--check" }
    if ($Mode) { $RunArgs += @("--mode", $Mode) }
    if ($Stuff) { $RunArgs += @("--stuff", ($Stuff -join ",")) }
    if ($Append) { $RunArgs += "--append" }
    if ($Clear) { $RunArgs += "--clear" }
    docker @DockerArgs --env-file $EnvFile $Image python /autolabel/cvat_run.py @RunArgs
}
exit $LASTEXITCODE
```

- [ ] **Step 4: Build và kiểm image**

```powershell
powershell -File autolabel\run.ps1 -Test
docker run --rm --gpus all cvat.sam3.sdk:2.74.1 python -c "import cvat_sdk, numpy, torch, sam3; from cvat_sdk.version import VERSION; print(VERSION, numpy.__version__, torch.cuda.is_available())"
```
Expected: lệnh đầu build image rồi in `Ran 0 tests` (chưa có test SDK; exit code 5 là bình thường ở bước này). Lệnh sau in `2.74.1 1.26.x True`.

- [ ] **Step 5: Commit**

```powershell
git add autolabel/Dockerfile.sdk autolabel/run.ps1
git commit -m "feat(autolabel): cvat-sdk GPU image and run.ps1 launcher"
```

---

## Task 3: `sam3_func.py` — SAM 3 dạng auto-annotation function

**Files:**
- Create: `autolabel/sam3_func.py`
- Test: `autolabel/tests/test_sdk_func.py` (chạy trong container)

**Interfaces:**
- Consumes: `core.load_processor`, `core.detect`, `core.postprocess`, `core.raw_record` (Task 1); test helpers `CFG`, `box`, `FakeProcessor` từ `autolabel/tests/test_core_sam3.py` (Task 1).
- Produces:
  - `sam3_func.ROOT: Path` (= thư mục `autolabel/`).
  - `sam3_func.Sam3Function(classes: list[str], mode: str, stuff: list[str], raw_dir: Path | str, task: str, processor=None, cfg: dict | None = None)`; thuộc tính `classes`, `mode`, `stuff`, `spec: DetectionFunctionSpec`, `counts: dict[str, int]`; `detect(context, image: PIL.Image) -> list[LabeledShapeRequest]`. `mode` sai → `ValueError`.
  - `sam3_func.create(classes: str, mode: str = "instance", stuff: str = "", task: str = "agent") -> Sam3Function` (tham số dạng chuỗi phân tách bằng dấu phẩy, cho `cvat-cli -p`).

- [ ] **Step 1: Viết test (fail)**

Tạo `autolabel/tests/test_sdk_func.py`:
```python
"""Sam3Function (cvat-sdk auto-annotation). Needs cvat-sdk: run with autolabel/run.ps1 -Test."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import cvat_sdk  # noqa: F401
except ImportError:
    raise unittest.SkipTest("cvat-sdk not installed; run autolabel/run.ps1 -Test")

from PIL import Image
from cvat_sdk.masks import decode_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sam3_func  # noqa: E402
from test_core_sam3 import CFG, FakeProcessor, box  # noqa: E402


def val(x):
    return getattr(x, "value", x)  # generated enum models wrap the string


def ctx(name):
    return SimpleNamespace(frame_name=name, conf_threshold=None, conv_mask_to_poly=False)


class Spec(unittest.TestCase):
    def test_labels_are_masks_in_class_order(self):
        fn = sam3_func.Sam3Function(["road", "car"], "instance", [], "unused", "t",
                                    processor=FakeProcessor({}), cfg=CFG)
        self.assertEqual([(l.name, l.id, val(l.type)) for l in fn.spec.labels],
                         [("road", 0, "mask"), ("car", 1, "mask")])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            sam3_func.Sam3Function(["car"], "boxes", [], "unused", "t", processor=FakeProcessor({}), cfg=CFG)


class Detect(unittest.TestCase):
    def test_masks_raw_and_counts(self):
        proc = FakeProcessor({"car": [(0.9, box(2, 5, 1, 4)), (0.8, box(6, 9, 6, 9))]})
        with tempfile.TemporaryDirectory() as d:
            fn = sam3_func.Sam3Function(["car"], "instance", [], d, "cp2_slice", processor=proc, cfg=CFG)
            shapes = fn.detect(ctx("sub/img7.jpg"), Image.new("RGB", (10, 10)))
            self.assertEqual([(s.label_id, val(s.type)) for s in shapes], [(0, "mask"), (0, "mask")])
            got = decode_mask(shapes[0].points, image_width=10, image_height=10)
            self.assertTrue((got == box(2, 5, 1, 4)).all())
            raw = json.loads((Path(d) / "img7.json").read_text())
            self.assertEqual((raw["task"], raw["image"], len(raw["objects"])), ("cp2_slice", "img7", 2))
            self.assertEqual(fn.counts, {"car": 2})

    def test_semantic_one_mask_per_class(self):
        proc = FakeProcessor({"road": [(0.8, box(0, 10, 0, 10))], "sky": [(0.9, box(0, 3, 0, 10))]})
        with tempfile.TemporaryDirectory() as d:
            fn = sam3_func.Sam3Function(["road", "sky"], "semantic", [], d, "t", processor=proc, cfg=CFG)
            shapes = fn.detect(ctx("a.jpg"), Image.new("RGB", (10, 10)))
            self.assertEqual(sorted(s.label_id for s in shapes), [0, 1])


class Create(unittest.TestCase):
    def test_factory_parses_comma_lists(self):
        fn = sam3_func.create(classes="road, sky,car", mode="panoptic", stuff="road,sky")
        self.assertEqual((fn.classes, fn.mode, fn.stuff), (["road", "sky", "car"], "panoptic", ["road", "sky"]))
        self.assertIsNone(fn.processor)  # model loads on the first frame only
```

- [ ] **Step 2: Chạy test, phải fail**

```powershell
powershell -File autolabel\run.ps1 -Test
```
Expected: ERROR `ModuleNotFoundError: No module named 'sam3_func'`.

- [ ] **Step 3: Viết `autolabel/sam3_func.py`**

```python
"""SAM 3 as a cvat-sdk auto-annotation (detection) function.

Used by cvat_run.py (annotate_task over the REST API). Also loadable by
`cvat-cli function create-native/run-agent --function-file autolabel/sam3_func.py -p classes=str:a,b -p mode=str:instance`.
Same SAM 3 prompts and post-processing as the nuclio function (nuclio/core.py, config.json).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import cvat_sdk.auto_annotation as cvataa
from cvat_sdk.masks import encode_mask

ROOT = Path(__file__).resolve().parent  # autolabel/: config.json, weights/, raw/
sys.path.insert(0, str(ROOT / "nuclio"))
import core  # noqa: E402

MODES = ("semantic", "instance", "panoptic")


class Sam3Function:
    def __init__(self, classes, mode, stuff, raw_dir, task, processor=None, cfg=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.classes, self.mode, self.stuff = list(classes), mode, list(stuff)
        self.raw_dir, self.task = Path(raw_dir), task
        self.cfg = cfg or json.loads((ROOT / "config.json").read_text())
        self.processor = processor  # loaded on the first frame
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[cvataa.label_spec(name, i, type="mask") for i, name in enumerate(self.classes)])
        self._ids = {name: i for i, name in enumerate(self.classes)}
        self.counts = {}

    def detect(self, context, image):
        t0 = time.time()
        if self.processor is None:
            self.processor = core.load_processor(ROOT / self.cfg["checkpoint"])
        img = image.convert("RGB")
        dets = core.detect(self.processor, img, self.classes, self.cfg)
        objs = core.postprocess(self.mode, dets, (img.height, img.width), self.stuff, self.cfg)
        stem = Path(context.frame_name).stem
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / f"{stem}.json").write_text(
            json.dumps(core.raw_record(self.task, stem, self.mode, self.cfg, time.time() - t0, objs)))
        for label, _, _ in objs:
            self.counts[label] = self.counts.get(label, 0) + 1
        print(f"{context.frame_name}: {len(objs)} objects in {time.time() - t0:.1f}s", flush=True)
        return [cvataa.mask(self._ids[label], encode_mask(np.asarray(m, bool))) for label, _, m in objs]


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def create(classes: str, mode: str = "instance", stuff: str = "", task: str = "agent"):
    """Factory for cvat-cli: -p classes=str:road,car -p mode=str:panoptic -p stuff=str:road."""
    return Sam3Function(_split(classes), mode, _split(stuff), ROOT / "raw" / "_online" / task, task)
```

- [ ] **Step 4: Chạy test, phải pass**

```powershell
powershell -File autolabel\run.ps1 -Test
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -m unittest discover -s autolabel/tests
```
Expected: trong container `Ran 5 tests` ... `OK`. Trên host `Ran 21 tests` ... `OK (skipped=1)` (module SDK tự skip, tính là 1 test skipped).

- [ ] **Step 5: Thử model thật trên 1 ảnh (tầng test 2)**

```powershell
docker run --rm --gpus all --mount "type=bind,source=$PWD\autolabel,target=/autolabel" --mount "type=bind,source=$PWD\data,target=/data,readonly" cvat.sam3.sdk:2.74.1 python -c "import sys,types; sys.path.insert(0,'/autolabel'); import sam3_func as f; from PIL import Image; fn=f.Sam3Function(['person','bicycle','car','motorcycle','bus','truck'],'instance',[],'/tmp/raw','cp2_slice'); s=fn.detect(types.SimpleNamespace(frame_name='000000017627.jpg'),Image.open('/data/checkpoints/cp2_slice/images/000000017627.jpg')); print(len(s), fn.counts)"
```
Expected: dòng `000000017627.jpg: N objects in X s` với N > 0, rồi `N {...}`; số theo class khớp bảng Day03 của ảnh này. (`/tmp/raw` nằm trong container nên không đè raw thật.)

- [ ] **Step 6: Commit**

```powershell
git add autolabel/sam3_func.py autolabel/tests/test_sdk_func.py
git commit -m "feat(autolabel): SAM 3 as a cvat-sdk auto-annotation function"
```

---

## Task 4: `cvat_run.py` — CLI chạy trên task qua API

**Files:**
- Create: `autolabel/cvat_run.py`
- Test: `autolabel/tests/test_sdk_run.py` (chạy trong container)

**Interfaces:**
- Consumes: `sam3_func.Sam3Function`, `sam3_func.ROOT` (Task 3); `core.task_classes(data_dir, task) -> (type, classes, stuff)` (có sẵn).
- Produces:
  - `cvat_run.resolve(task_id: int, task_name: str, task_labels: list[str], data_dir: Path, mode: str | None = None, stuff: list[str] | None = None) -> tuple[list[str], str, list[str], str, list[str]]` = `(classes, mode, stuff, raw_subdir, warnings)`; raise `ValueError` khi không có class hoặc panoptic thiếu stuff.
  - `cvat_run.guard(n_shapes: int, n_tracks: int, n_tags: int, append: bool, clear: bool) -> str | None`.
  - `cvat_run.parse_args(argv: list[str]) -> argparse.Namespace` (`task`, `mode`, `stuff`, `check`, `append`, `clear`).
  - `cvat_run.main(argv=None) -> int` (0 ok, 1 lỗi API/mạng, 2 lỗi dùng sai/dừng an toàn).
  - Env: `CVAT_URL`, `CVAT_ACCESS_TOKEN`, `CVAT_ORG` (rỗng = không có), `AUTOLABEL_DATA` (mặc định `/data`).

- [ ] **Step 1: Viết test (fail)**

Tạo `autolabel/tests/test_sdk_run.py`:
```python
"""cvat_run.py pure logic (class/mode choice, safety guard, args). Run with autolabel/run.ps1 -Test."""
import sys
import unittest
from pathlib import Path

try:
    import cvat_sdk  # noqa: F401
except ImportError:
    raise unittest.SkipTest("cvat-sdk not installed; run autolabel/run.ps1 -Test")

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"  # repo data/ on the host, /data in the container
sys.path.insert(0, str(HERE.parent))
import cvat_run  # noqa: E402

VEHICLES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]  # cp2_slice classes.json
STUFF = ["road", "sidewalk", "building", "vegetation", "sky"]


class Resolve(unittest.TestCase):
    def test_day03_instance_task_uses_classes_json(self):
        classes, mode, stuff, raw, warns = cvat_run.resolve(5, "cp2_slice", VEHICLES + ["extra"], DATA)
        self.assertEqual((classes, mode, stuff, raw), (VEHICLES, "instance", [], "cp2_slice"))
        self.assertEqual(len(warns), 1)
        self.assertIn("extra", warns[0])

    def test_day03_panoptic_gets_stuff(self):
        labels = STUFF + ["person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"]
        classes, mode, stuff, raw, warns = cvat_run.resolve(7, "hard_panoptic", labels, DATA)
        self.assertEqual((mode, stuff, raw, warns), ("panoptic", STUFF, "hard_panoptic", []))
        self.assertEqual(len(classes), 12)

    def test_missing_day03_label_warns(self):
        _, _, _, _, warns = cvat_run.resolve(5, "cp4_curb", ["road"], DATA)
        self.assertIn("sidewalk", warns[0])

    def test_other_task_uses_task_labels(self):
        self.assertEqual(cvat_run.resolve(42, "street-batch-1", ["car", "road"], DATA),
                         (["car", "road"], "instance", [], "_online/42", []))

    def test_overrides(self):
        classes, mode, stuff, _, _ = cvat_run.resolve(42, "x", ["road", "car"], DATA, "panoptic", ["road"])
        self.assertEqual((mode, stuff), ("panoptic", ["road"]))
        self.assertEqual(cvat_run.resolve(5, "cp2_slice", VEHICLES, DATA, mode="semantic")[1], "semantic")

    def test_errors(self):
        with self.assertRaises(ValueError):
            cvat_run.resolve(42, "x", ["road", "car"], DATA, "panoptic")  # no --stuff
        with self.assertRaises(ValueError):
            cvat_run.resolve(5, "cp2_slice", ["sky"], DATA)  # no Day03 class in the task


class Guard(unittest.TestCase):
    def test_existing_annotations_need_a_flag(self):
        self.assertIsNone(cvat_run.guard(0, 0, 0, False, False))
        self.assertIn("--append", cvat_run.guard(3, 0, 0, False, False))
        self.assertIn("--clear", cvat_run.guard(0, 1, 0, False, False))
        self.assertIsNone(cvat_run.guard(3, 0, 0, True, False))
        self.assertIsNone(cvat_run.guard(3, 0, 0, False, True))


class Args(unittest.TestCase):
    def test_stuff_list_and_flags(self):
        a = cvat_run.parse_args(["--task", "12", "--stuff", "road, sky", "--check"])
        self.assertEqual((a.task, a.stuff, a.check, a.append, a.clear), (12, ["road", "sky"], True, False, False))

    def test_append_and_clear_are_exclusive(self):
        with self.assertRaises(SystemExit):
            cvat_run.parse_args(["--task", "1", "--append", "--clear"])
```

- [ ] **Step 2: Chạy test, phải fail**

```powershell
powershell -File autolabel\run.ps1 -Test
```
Expected: ERROR `ModuleNotFoundError: No module named 'cvat_run'`.

- [ ] **Step 3: Viết `autolabel/cvat_run.py`**

```python
"""Run SAM 3 on a CVAT task through the REST API (cvat-sdk annotate_task), authenticated with a PAT.

Env: CVAT_URL, CVAT_ACCESS_TOKEN (Personal Access Token, read/write), optional CVAT_ORG.
Started by autolabel/run.ps1, which mounts autolabel/ at /autolabel and data/ at /data.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import urllib3
from cvat_sdk import make_client
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.auto_annotation import annotate_task

from sam3_func import ROOT, Sam3Function

sys.path.insert(0, str(ROOT / "nuclio"))
import core  # noqa: E402

DATA = Path(os.environ.get("AUTOLABEL_DATA", "/data"))
HINTS = {401: "token invalid or expired: create a new Personal Access Token",
         403: "no permission: check the task id and CVAT_ORG",
         404: "task not found: check the task id and CVAT_ORG"}


def resolve(task_id, task_name, task_labels, data_dir, mode=None, stuff=None):
    """(classes, mode, stuff, raw_subdir, warnings). Day03 tasks (name in data/manifest.json) follow classes.json."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())["tasks"]
    warnings = []
    if task_name in manifest:
        ttype, day3, day3_stuff = core.task_classes(data_dir, task_name)
        classes = [c for c in day3 if c in task_labels]
        if set(day3) - set(task_labels):
            warnings.append(f"task has no label for Day03 classes {sorted(set(day3) - set(task_labels))}")
        if set(task_labels) - set(day3):
            warnings.append(f"skipping task labels not in classes.json: {sorted(set(task_labels) - set(day3))}")
        mode, stuff, raw_sub = mode or ttype, day3_stuff if stuff is None else stuff, task_name
    else:
        classes, mode, stuff, raw_sub = list(task_labels), mode or "instance", stuff or [], f"_online/{task_id}"
    if not classes:
        raise ValueError(f"no class to annotate; task labels: {sorted(task_labels)}")
    if mode == "panoptic" and not stuff:
        raise ValueError("panoptic needs --stuff (comma-separated stuff classes, e.g. road,sky)")
    if set(stuff) - set(classes):
        warnings.append(f"--stuff names not among the classes: {sorted(set(stuff) - set(classes))}")
    return classes, mode, list(stuff), raw_sub, warnings


def guard(n_shapes, n_tracks, n_tags, append, clear):
    """Message when a run would silently stack on existing work, else None."""
    if (n_shapes or n_tracks or n_tags) and not (append or clear):
        return (f"task already has {n_shapes} shapes, {n_tracks} tracks, {n_tags} tags; rerun with "
                "-Append/--append (keep them, add SAM 3 masks) or -Clear/--clear (DELETE them first)")
    return None


def _list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run SAM 3 on a CVAT task via the REST API.")
    p.add_argument("--task", type=int, required=True, help="CVAT task id (the number in .../tasks/<id>)")
    p.add_argument("--mode", choices=["semantic", "instance", "panoptic"],
                   help="default: classes.json for Day03 tasks, else instance")
    p.add_argument("--stuff", type=_list, help="stuff classes for panoptic, e.g. road,sky")
    p.add_argument("--check", action="store_true", help="read-only: show what would happen, download/write nothing")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--append", action="store_true", help="keep existing annotations and add SAM 3 masks")
    g.add_argument("--clear", action="store_true", help="DELETE all existing annotations of the task first")
    return p.parse_args(argv)


def run(client, args):
    task = client.tasks.retrieve(args.task)
    if task.mode != "annotation":
        raise ValueError(f"task {args.task} is a video task (mode={task.mode}); only image tasks are supported")
    labels = [label.name for label in task.get_labels()]
    classes, mode, stuff, raw_sub, warnings = resolve(args.task, task.name, labels, DATA, args.mode, args.stuff)
    ann = task.get_annotations()
    n = (len(ann.shapes), len(ann.tracks), len(ann.tags))
    print(f"server {client.get_server_version()} | task {args.task} '{task.name}' | {task.size} frames | "
          f"existing: {n[0]} shapes, {n[1]} tracks, {n[2]} tags")
    print(f"mode {mode} | classes {classes} | stuff {stuff} | raw -> autolabel/raw/{raw_sub}/")
    for w in warnings:
        print(f"warning: {w}")
    if args.check:
        _, resp = client.api_client.call_api("/api/functions", "GET", _parse_response=False, _check_status=False)
        print("AI agents (/api/functions): " + ("available" if resp.status == 200 else f"not available (HTTP {resp.status})"))
        return 0
    err = guard(*n, args.append, args.clear)
    if err:
        print(f"stopped: {err}", file=sys.stderr)
        return 2
    if args.clear:
        print(f"--clear: deleting {n[0]} shapes, {n[1]} tracks, {n[2]} tags of task {args.task}")
    fn = Sam3Function(classes, mode, stuff, ROOT / "raw" / raw_sub, task.name)
    t0 = time.time()
    annotate_task(client, args.task, fn, clear_existing=args.clear, allow_unmatched_labels=True)
    print(f"done in {time.time() - t0:.0f}s, uploaded once at the end: {fn.counts}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    url, token = os.environ.get("CVAT_URL"), os.environ.get("CVAT_ACCESS_TOKEN")
    org = os.environ.get("CVAT_ORG") or None
    if not url or not token:
        print("CVAT_URL and CVAT_ACCESS_TOKEN must be set (autolabel/.env.<target>)", file=sys.stderr)
        return 2
    try:
        with make_client(url, access_token=token) as client:
            if org:
                client.organization_slug = org
            return run(client, args)
    except ApiException as e:
        print(f"CVAT API error {e.status}: {HINTS.get(e.status, e.reason)}", file=sys.stderr)
        return 1
    except urllib3.exceptions.HTTPError as e:
        print(f"Cannot reach CVAT at {url}: {e}. Check CVAT_URL / VPN.", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Chạy test, phải pass**

```powershell
powershell -File autolabel\run.ps1 -Test
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -m unittest discover -s autolabel/tests
```
Expected: trong container `Ran 14 tests` ... `OK`. Trên host `Ran 22 tests` ... `OK (skipped=2)`.

- [ ] **Step 5: Commit**

```powershell
git add autolabel/cvat_run.py autolabel/tests/test_sdk_run.py
git commit -m "feat(autolabel): cvat_run CLI - SAM 3 on a CVAT task via REST API + PAT"
```

---

## Task 5: `.env`, `.gitignore`, README và kiểm đường lỗi

**Files:**
- Create: `autolabel/.env.example`
- Modify: `.gitignore` (thêm 3 dòng cuối)
- Modify: `autolabel/README.md` (bảng "Cấu trúc" + mục mới)

**Interfaces:**
- Consumes: `run.ps1` (Task 2), `cvat_run.py` (Task 4).
- Produces: `autolabel/.env.local` / `autolabel/.env.online` (học viên tự tạo từ `.env.example`, bị ignore).

- [ ] **Step 1: Tạo `autolabel/.env.example`**

```
# Copy to autolabel/.env.local (CVAT local) or autolabel/.env.online (program's CVAT). Never commit the copies.
# CVAT local (container joins network cvat_cvat): http://cvat_server:8080 ; online: https://<server>
CVAT_URL=http://cvat_server:8080
# Personal Access Token: CVAT avatar -> Settings -> Security -> Personal access tokens (read/write, short expiry)
CVAT_ACCESS_TOKEN=paste-your-token-here
# Organization slug if the task lives in an organization, else leave empty
CVAT_ORG=
```

- [ ] **Step 2: Thêm vào cuối `.gitignore`**

```
autolabel/.env*
!autolabel/.env.example
autolabel/raw/_online/
```
Kiểm:
```powershell
git check-ignore -v autolabel/.env.local autolabel/.env.online autolabel/raw/_online/42/a.json
git check-ignore autolabel/.env.example; "exit=$LASTEXITCODE"
```
Expected: 3 dòng đầu đều bị ignore; `.env.example` in `exit=1` (không bị ignore).

- [ ] **Step 3: Kiểm đường lỗi (CVAT local đang chạy, token giả)**

```powershell
Copy-Item autolabel\.env.example autolabel\.env.local
powershell -File autolabel\run.ps1 -Target local -Task 1 -Check; "exit=$LASTEXITCODE"
powershell -File autolabel\run.ps1 -Target nope -Task 1
```
Expected: lệnh 1 in `CVAT API error 401: token invalid or expired: create a new Personal Access Token`, `exit=1`, **không** in token. Lệnh 2 báo `Missing ...\.env.nope - copy autolabel\.env.example ...`.

- [ ] **Step 4: README**

Trong bảng `## Cấu trúc` của `autolabel/README.md`, thêm sau dòng `| \`tests/\` | Test không cần weights |`:
```markdown
| `sam3_func.py`, `cvat_run.py` | SAM 3 dạng auto-annotation function của `cvat-sdk` và CLI chạy nó trên task qua REST API + PAT (CVAT không deploy được nuclio) |
| `Dockerfile.sdk`, `run.ps1`, `.env.example` | Image GPU + `cvat-sdk`, script chạy, mẫu biến môi trường |
```
Thêm mục mới ngay trước `## Test (không cần weights)`:
````markdown
## Chạy trên CVAT không deploy được nuclio (CVAT online, qua API + PAT)

Thiết kế: [`docs/superpowers/specs/2026-11-12-cvat-online-sam3-design.md`](../docs/superpowers/specs/2026-11-12-cvat-online-sam3-design.md). Tool chạy trong container GPU trên máy bạn, tải frame của task qua REST API, chạy SAM 3, ghi mask vào task. Đăng nhập bằng Personal Access Token (PAT). Frame chỉ nằm trong cache của container và mất khi container dừng.

1. **Tạo PAT** trên CVAT: avatar → **Settings → Security → Personal access tokens**, quyền **read/write**, hạn ngắn (vài ngày). Token chỉ hiện một lần.
2. Chép `autolabel/.env.example` thành `autolabel/.env.local` (CVAT local, `CVAT_URL=http://cvat_server:8080`) hoặc `autolabel/.env.online` (CVAT chương trình), điền `CVAT_URL`, `CVAT_ACCESS_TOKEN`, `CVAT_ORG` nếu task nằm trong organization. Hai file này bị git-ignore. Không dán token vào chat, không commit.
3. **Kiểm trước** (không tải ảnh, không ghi gì):

   ```powershell
   powershell -File autolabel\run.ps1 -Target online -Task 123 -Check
   ```

   In version server, task, label, số annotation đang có, mode/class sẽ dùng, server có AI agent hay không. Nếu SDK cảnh báo lệch version, thêm `-SdkVersion <version server>` (tự build image mới).
4. **Chạy:**

   ```powershell
   powershell -File autolabel\run.ps1 -Target online -Task 123
   ```

   - Task Day03 (tên task trùng `data/manifest.json`): class và mode lấy từ `classes.json`.
   - Task khác: class = label của task, mode mặc định `instance`; đổi bằng `-Mode semantic|panoptic`; panoptic cần `-Stuff road,sky`.
   - Task đã có annotation → tool dừng. `-Append` giữ và thêm; `-Clear` **xoá hết** annotation của task rồi ghi mới.
   - SDK chỉ upload một lần ở cuối: lỗi giữa chừng thì task không đổi.
5. Mask gốc: task Day03 → `autolabel/raw/<task>/`; task khác → `autolabel/raw/_online/<id>/` (git-ignore).
6. Xong việc: thu hồi PAT trên CVAT, xoá `autolabel/.env.online`.

Test cần `cvat-sdk` chạy trong container: `powershell -File autolabel\run.ps1 -Test`.
````

- [ ] **Step 5: Commit**

```powershell
git status --short   # KHÔNG được thấy .env.local
git add .gitignore autolabel/.env.example autolabel/README.md
git commit -m "docs(autolabel): PAT setup, env template and online usage"
```

---

## Task 6: Diễn tập Day03 trên CVAT local với PAT (tầng test 3)

Cần học viên làm các bước có ký hiệu **[HV]** (tài khoản CVAT, token). Executor chạy lệnh và báo kết quả.

**Files:** không sửa code. Nếu lộ bug, quay về task tương ứng (test trước, sửa sau).

- [ ] **Step 1 [HV]: PAT local**

Trên http://localhost:8080: avatar → **Settings → Security → Personal access tokens** → tạo token read/write, hạn 3 ngày. Mở `autolabel\.env.local`, thay `paste-your-token-here` bằng token. Giữ `CVAT_URL=http://cvat_server:8080`.

- [ ] **Step 2 [HV]: Task test riêng**

**Tasks → Create new task**: Name `cp2_slice` (trùng tên Day03; CVAT cho trùng tên), ảnh `data/checkpoints/cp2_slice/images/000000017627.jpg`, Labels → Raw dán `data/checkpoints/cp2_slice/cvat-labels.json`. Ghi lại **ID** trong URL `.../tasks/<ID>`. Task này khác task `cp2_slice` đang làm bài.

- [ ] **Step 3: `--check`**

```powershell
powershell -File autolabel\run.ps1 -Target local -Task <ID> -Check
```
Expected: `server 2.74.1 | task <ID> 'cp2_slice' | 1 frames | existing: 0 shapes, 0 tracks, 0 tags`; `mode instance | classes ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'] | stuff [] | raw -> autolabel/raw/cp2_slice/`; `AI agents (/api/functions): not available (HTTP 404)` (hoặc mã khác 200).

- [ ] **Step 4: Chạy thật và so với nuclio**

```powershell
Copy-Item autolabel\raw\cp2_slice\000000017627.json "$env:TEMP\nuclio_cp2.json"
powershell -File autolabel\run.ps1 -Target local -Task <ID>
$env:PYTHONUTF8 = "1"; & "$HOME\.venvs\day03\Scripts\python.exe" -c "import json,sys,collections as c;f=lambda p:c.Counter(o['label'] for o in json.load(open(p))['objects']);a,b=f(sys.argv[1]),f(sys.argv[2]);print(a,b);sys.exit(a!=b)" "$env:TEMP\nuclio_cp2.json" autolabel\raw\cp2_slice\000000017627.json
```
Expected: `000000017627.jpg: N objects in X s`, `done in ...s, uploaded once at the end: {...}`; hai `Counter` giống nhau (exit 0). Lệch → kiểm frame nén / class / mode trước khi đi tiếp.

- [ ] **Step 5 [HV]: Xem trong UI**

Mở job của task `<ID>`: thấy N mask, đúng label (ví dụ `car`), biên hợp lý.

- [ ] **Step 6: Cơ chế bảo vệ**

```powershell
powershell -File autolabel\run.ps1 -Target local -Task <ID>; "exit=$LASTEXITCODE"
powershell -File autolabel\run.ps1 -Target local -Task <ID> -Append
powershell -File autolabel\run.ps1 -Target local -Task <ID> -Check
powershell -File autolabel\run.ps1 -Target local -Task <ID> -Clear
powershell -File autolabel\run.ps1 -Target local -Task <ID> -Check
```
Expected: lần 1 `stopped: task already has N shapes ...`, `exit=2`. Sau `-Append`, check in `existing: 2N shapes`. `-Clear` in `--clear: deleting 2N shapes ...`; check sau đó in `existing: N shapes`.

- [ ] **Step 7 [HV]: Dọn**

Xoá task test **theo ID `<ID>`** (Actions → Delete), cẩn thận không xoá task `cp2_slice` làm bài. Raw `autolabel/raw/cp2_slice/000000017627.json` giờ là bản SDK (cùng số object với bản nuclio), giữ được.

---

## Task 7: CVAT online (tầng test 4 và 5)

**Files:** không sửa code (trừ khi phải đổi `-SdkVersion` mặc định — xem Step 3).

- [ ] **Step 1 [HV]: Điều kiện**

Chương trình đã xác nhận được tải ảnh của task về máy học viên qua API để chạy model. Chưa có xác nhận → **dừng ở Step 3** (`--check` không tải ảnh).

- [ ] **Step 2 [HV]: PAT online**

Trên CVAT chương trình: tạo PAT read/write, hạn khoảng 3 ngày. `Copy-Item autolabel\.env.example autolabel\.env.online`, điền `CVAT_URL=https://<server>`, token, `CVAT_ORG` nếu task nằm trong organization.

- [ ] **Step 3: `--check` online**

```powershell
powershell -File autolabel\run.ps1 -Target online -Task <ID online> -Check
```
Expected: in `server X.Y.Z | task ...`. Có cảnh báo SDK `Server version 'X.Y.Z' is not compatible with SDK version '2.74.1'` → chạy lại với `-SdkVersion X.Y.Z` (nếu bản đó có trên PyPI; xem Task 2 Step 1). Lỗi 401/403/404 → làm theo gợi ý in ra. Ghi lại dòng `AI agents (...)`.

- [ ] **Step 4: Quyết phương án B**

`AI agents: available` → báo học viên; agent là plan riêng (dùng `sam3_func.create`). Không thì bỏ B.

- [ ] **Step 5 [HV + executor]: Chạy thật lần đầu**

Chỉ khi Step 1 đạt. Chọn 1 task nhỏ được phép, chưa có annotation:
```powershell
powershell -File autolabel\run.ps1 -Target online -Task <ID online> -Check
powershell -File autolabel\run.ps1 -Target online -Task <ID online>
```
Task không phải Day03: thêm `-Mode ...` (và `-Stuff ...` nếu panoptic) theo loại bài. Học viên xem kết quả trong UI rồi mới chạy task khác.

- [ ] **Step 6 [HV]: Dọn sau khi xong**

Thu hồi PAT online và PAT local trên CVAT; xoá `autolabel\.env.online` và `autolabel\.env.local`.

---

## Self-review (đã chạy)

- Spec coverage: kiến trúc/thành phần → Task 1–5; luồng 1–7 → Task 4 (`run`, `resolve`, `guard`, `--check`); lỗi (401/403/404, video, mạng, không class, thiếu weights) → Task 4 `main`/`run`, Task 1 `load_processor`; bảo mật (`--env-file`, ignore `.env*`, `raw/_online/`, cache container) → Task 2, 5; test tầng 1–5 → Task 1/3/4, Task 3 Step 5, Task 6, Task 7; phương án B → Task 3 `create` + Task 7 Step 4; điều kiện chạy dữ liệu thật → Task 7 Step 1.
- Tên dùng xuyên task: `Sam3Function(classes, mode, stuff, raw_dir, task, processor, cfg)`, `create(classes, mode, stuff, task)`, `resolve(...) -> (classes, mode, stuff, raw_sub, warnings)`, `guard(n_shapes, n_tracks, n_tags, append, clear)`, image `cvat.sam3.sdk:<version>`, `run.ps1 -Target/-SdkVersion/-Test`.
- Số test: host 17 → 20 (Task 1), SDK module tự skip trên host; container 5 (Task 3) → 14 (Task 4); host 21 rồi 22 (mỗi module SDK skip tính 1).
