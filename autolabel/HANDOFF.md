# Handoff: chạy SAM 3 auto-label trên máy có GPU

Ngày: 2026-11-11 · Hạn nộp bài Day03: **12/11/2026 12:00 (UTC+7)**.

Tài liệu liên quan: [README.md](README.md) (cách dùng chi tiết) · [thiết kế](../docs/superpowers/specs/2026-11-11-sam3-autolabel-design.md).

## 1. Đang ở đâu

| Hạng mục | Trạng thái |
| --- | --- |
| Code `autolabel/` (function nuclio, hậu xử lý, `compare.py`, config) | Viết xong. 17 unit test pass (`python3 -m unittest discover -s autolabel/tests`) |
| Bản GPU (`nuclio/Dockerfile.gpu`, `nuclio/function-gpu.yaml`, `deploy.sh gpu`, `main.py` tự chọn `cuda`) | Viết xong, **chưa chạy lần nào** |
| Weights `facebook/sam3` → `sam3.pt` (3.45 GB) | Đã có quyền truy cập, đã tải về máy cũ (`autolabel/weights/sam3.pt`, git-ignore) |
| Image Docker | **Chưa build thành công lần nào** (máy cũ mạng rớt giữa chừng) |
| Inference SAM 3 thật | **Chưa chạy lần nào**: chưa biết tốc độ, chưa biết chất lượng |
| Máy cũ (CPU) | Nuclio đã bật, `nuctl` 1.16.3 ở `~/.local/bin`, timeout CVAT đã nâng lên 900 s |

Code chưa được commit, chỉ nằm trên ổ máy cũ: `autolabel/`, `docs/superpowers/`, và một dòng thêm vào `.gitignore` (`autolabel/weights/`).

## 2. Mang sang máy GPU

Chọn một trong hai cách:

- **Qua fork (khuyên dùng):** trên máy cũ commit và push `autolabel/`, `docs/superpowers/`, `.gitignore` lên fork, rồi clone fork trên máy GPU. Weights không đi theo git (đã ignore).
- **Copy tay:** chép cả thư mục repo, kể cả `autolabel/weights/sam3.pt` nếu muốn khỏi tải lại 3.4 GB.

**Annotation đang có trong CVAT máy cũ** không tự đi theo. Nếu đã vẽ gì trên máy cũ, mở task → **Actions → Backup task**, rồi trên máy GPU vào **Tasks → Create from backup**.

## 3. Yêu cầu của máy GPU

| Thành phần | Yêu cầu | Kiểm tra |
| --- | --- | --- |
| OS | Linux x86_64 (Ubuntu 22.04/24.04). Windows thì dùng WSL2 + Docker Desktop có GPU | `uname -m` → `x86_64` |
| GPU | NVIDIA, VRAM nên ≥ 8 GB (model 848M tham số, ~3.4 GB fp32; *con số này là ước tính, chưa đo*) | `nvidia-smi` |
| Driver NVIDIA | Đủ mới cho CUDA 12.8 (driver ≥ 570) | `nvidia-smi` dòng "CUDA Version" ≥ 12.8 |
| Docker + Compose v2 | Docker Engine 24+ | `docker compose version` |
| NVIDIA Container Toolkit | Để container thấy GPU | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| `nuctl` | **Đúng 1.16.3** (khớp dashboard nuclio của CVAT v2.74.1) | `nuctl version` |
| Python | 3.10+ trên host (cho `compare.py`, `hf`) | `python3 --version` |
| Ổ đĩa | Khoảng 25 GB trống (CVAT ~5 GB, image torch CUDA ~10 GB, weights 3.4 GB) | `df -h` |
| Mạng | Ổn định khi build: tải torch CUDA khoảng 3 GB | |

Cài những gì còn thiếu:

```bash
# NVIDIA Container Toolkit (Ubuntu), theo hướng dẫn chính thức của NVIDIA:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# nuctl 1.16.3
mkdir -p ~/.local/bin
curl -fL -o ~/.local/bin/nuctl https://github.com/nuclio/nuclio/releases/download/1.16.3/nuctl-1.16.3-linux-amd64
chmod +x ~/.local/bin/nuctl && export PATH="$HOME/.local/bin:$PATH"

# Hugging Face CLI (venv riêng, không đụng Python hệ thống)
python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install -U huggingface_hub
```

## 4. Các bước chạy

Đặt biến cho gọn (sửa đường dẫn theo máy):

```bash
REPO=~/K5-L2-Day03-ChuThienPhu-2A202603114-Image-Segmentation
CVAT=~/cvat
```

1. **CVAT v2.74.1 kèm serverless** (máy GPU chưa có CVAT):

   ```bash
   git clone --depth 1 --branch v2.74.1 https://github.com/cvat-ai/cvat.git "$CVAT"
   cd "$CVAT"
   docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml \
     -f "$REPO/autolabel/cvat-timeout.yml" up -d
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```

   Mở http://localhost:8080 và đăng nhập. Nếu máy đã có CVAT thì chỉ chạy lệnh `up -d` ở trên. **Không bao giờ `down -v`**, vì lệnh đó xoá dữ liệu.

2. **Weights** (bỏ qua nếu đã copy `sam3.pt` sang):

   ```bash
   ~/.venvs/hf/bin/hf auth login          # dán token Read; KHÔNG lưu token vào repo
   cd "$REPO" && PATH="$HOME/.venvs/hf/bin:$PATH" autolabel/download_weights.sh
   ```

3. **Deploy bản GPU:**

   ```bash
   cd "$REPO" && autolabel/deploy.sh gpu
   ```

   Function giữ nguyên tên `sam3-autolabel`, nên deploy bản GPU sẽ thay thế bản CPU nếu đã có.

4. **Thử 1 ảnh, chưa qua CVAT:**

   ```bash
   cd "$REPO"
   PORT=$(docker port nuclio-nuclio-sam3-autolabel 8080 | head -1 | cut -d: -f2)
   python3 -c "import base64,json;print(json.dumps({'image':base64.b64encode(open('data/checkpoints/cp2_slice/images/000000017627.jpg','rb').read()).decode()}))" \
     | curl -s -X POST -H 'Content-Type: application/json' --data @- "http://localhost:$PORT" | head -c 300; echo
   docker logs --tail 20 nuclio-nuclio-sam3-autolabel   # có "image -> task=cp2_slice" và "N objects in X s"
   ls autolabel/raw/cp2_slice/                           # có 000000017627.json
   ```

   Kiểm tra function thật sự dùng GPU: chạy `nvidia-smi` trong lúc gọi, phải thấy tiến trình python chiếm VRAM.

5. **Chạy trong CVAT** cho 9 task: tạo task theo README chính, rồi **Actions → Automatic annotation → SAM 3 auto-label (Day03)**. Sau đó sửa tay, Save, export ZIP vào `submissions/<task>.zip`.

6. **Đo phần do tool làm:**

   ```bash
   cd "$REPO" && pip install -r requirements.txt && python3 autolabel/compare.py
   ```

   Lệnh ghi bảng vào `autolabel/AUTOLABEL_REPORT.md`. Khi có ground truth: `python3 autolabel/compare.py --gt <thư mục reference nằm NGOÀI repo>`.

## 5. Chỉnh chất lượng (sau lần chạy đầu)

Chỉ cần sửa `autolabel/config.json`, không phải deploy lại; lần gọi kế tiếp tự dùng giá trị mới.

- Class bị sót (thường gặp với vùng nền như `sidewalk`, `building`): hạ ngưỡng, ví dụ `"thresholds": {"sidewalk": 0.25}`, hoặc thêm prompt đồng nghĩa như `"sidewalk": ["sidewalk", "pavement"]`.
- Class bị nhận nhầm: nâng ngưỡng của class đó.
- Mảnh vụn: tăng `min_area`. Hai xe bị gộp hoặc trùng nhau: chỉnh `nms_iou`.
- Rủi ro đã biết: SAM 3 mạnh với vật đếm được, yếu hơn với vùng nền. Ranh road–sidewalk (`cp4_curb`, `easy_semantic`) nhiều khả năng vẫn phải sửa tay.

## 6. Rủi ro và điểm chưa kiểm chứng

1. **Chưa từng chạy SAM 3 thật.** Các lệnh `build_sam3_image_model(checkpoint_path=..., load_from_HF=False, device=...)` và `Sam3Processor(...)` viết theo source upstream ở commit `f92a5ef`. Nếu lỗi tham số, xem `sam3/model_builder.py` trong image.
2. **Định dạng mask trả cho CVAT** viết theo `to_cvat_mask` của chính CVAT v2.74.1 (subagent đối chiếu) và đã có unit test, nhưng chưa thử qua CVAT thật. Nếu Automatic annotation chạy xong mà không thấy mask, xem log `cvat_worker_annotation`.
3. **Bản GPU chưa build lần nào.** Bản CPU có vá 2 chỗ `device="cuda"` cố định; bản GPU không cần vá nên không có bước đó.
4. Function khớp ảnh với `data/` qua thumbnail. Nếu task tạo từ ảnh khác bộ 14 ảnh, kết quả rơi vào `raw/unknown/` và chạy đủ 14 class.
5. Phần "50%+" do giảng viên đánh giá. `compare.py` chỉ đưa ra con số làm bằng chứng.

## 7. Không được làm

- Commit `autolabel/weights/`, token HF, hoặc ground truth (RULES.md; test repo cấm mọi thư mục `groundtruth`).
- Sửa `data/`, `tests/`, `notebooks/`, `.github/`; đặt ZIP lạ trong `submissions/`.
- `docker compose down -v` trên CVAT.
- Thêm dòng `Co-Authored-By` hay ghi công AI vào commit message (quy ước riêng của học viên).
