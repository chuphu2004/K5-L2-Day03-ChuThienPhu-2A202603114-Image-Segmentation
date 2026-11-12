# SAM 3 auto-label cho CVAT (Day03 bonus)

Function **nuclio** chạy SAM 3 (`facebook/sam3`, pretrained, CPU) bằng text prompt cho từng class của task, trả mask về CVAT qua **Actions → Automatic annotation**, đồng thời lưu **mask gốc** của tool vào `autolabel/raw/` để sau này đo xem bao nhiêu phần bài nộp là do tool làm.

Thiết kế: [`docs/superpowers/specs/2026-11-11-sam3-autolabel-design.md`](../docs/superpowers/specs/2026-11-11-sam3-autolabel-design.md).

## Cấu trúc

| File | Vai trò |
| --- | --- |
| `nuclio/function.yaml` | Khai báo function (detector, 14 label trùng tên `classes.json`), timeout 600 s |
| `nuclio/Dockerfile` | Image nền CPU: Python 3.12, torch 2.10 CPU, `sam3` (pin commit), vá CUDA |
| `nuclio/main.py` | Handler: nhận ảnh → nhận diện task → SAM 3 → hậu xử lý → trả CVAT + lưu raw |
| `nuclio/core.py` | Phần thuần numpy: khớp thumbnail, hậu xử lý semantic/instance/panoptic, mask CVAT, RLE |
| `config.json` | Prompt và ngưỡng theo class, đường dẫn weights. **Sửa xong có hiệu lực ngay ở lần gọi sau**, không cần deploy lại |
| `cvat-timeout.yml` | Compose override nâng timeout CVAT → nuclio lên 900 s (mặc định 120 s, quá ngắn cho CPU) |
| `download_weights.sh` | Tải `sam3.pt` về `autolabel/weights/` (git-ignore) |
| `deploy.sh` | Build image nền + deploy function vào nuclio của CVAT |
| `compare.py` | So mask gốc với ZIP trong `submissions/`, ghi `AUTOLABEL_REPORT.md` |
| `raw/<task>/<ảnh>.json` | Mask gốc mỗi lần gọi (tạo tự động, gọi lại thì ghi đè) |
| `tests/` | Test không cần weights |

## Cách function hoạt động

- CVAT chỉ gửi ảnh (base64), không gửi tên file hay task. Function so thumbnail 32×32 xám của ảnh nhận được với 14 ảnh trong `data/` (khoảng cách tối đa `match_max_dist` = 12; ảnh khác nhau trong bộ data cách nhau ≥ 38, ảnh sau khi CVAT nén lại cách bản gốc < 1). Từ đó biết task và đọc `classes.json`. Không khớp → chạy tất cả 14 class kiểu instance, lưu raw vào `raw/unknown/`.
- Một lần `set_image`, mỗi class chạy từng prompt trong `config.json` (`prompts`), giữ mask có score ≥ `thresholds[class]` (nếu không có thì `default_threshold`).
- Hậu xử lý theo loại task:
  - `instance`: bỏ mask < `min_area` pixel, NMS theo mask IoU (`nms_iou`) trong cùng class.
  - `semantic`: mỗi pixel thuộc class có score cao nhất; mỗi class một mask.
  - `panoptic`: stuff như semantic, thing như instance, thing đè lên stuff.
- Ngưỡng (threshold) nhập trong hộp thoại Automatic annotation của CVAT **không được dùng**; chỉnh trong `config.json`.

## Chạy lần đầu

Các bước 1–3 chỉ làm một lần. Lệnh chạy từ gốc repo này, trừ khi ghi khác.

1. **Xin quyền weights**: đăng nhập Hugging Face, mở https://huggingface.co/facebook/sam3, điền form xin truy cập, chờ duyệt.
2. **Đăng nhập và tải weights** (token chỉ nằm trong `~/.cache/huggingface` trên máy, không vào repo, không vào image):

   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login
   autolabel/download_weights.sh        # ~3.4 GB -> autolabel/weights/sam3.pt
   ```

3. **Bật serverless cho CVAT đang chạy** (giữ nguyên dữ liệu; tuyệt đối không `down -v`):

   ```bash
   cd /home/phu/Documents/AI20K/cvat-day2
   docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml \
     -f "<đường dẫn repo>/autolabel/cvat-timeout.yml" up -d
   ```

   Cần `nuctl` 1.16.3 (cùng phiên bản dashboard nuclio) trong `PATH`, ví dụ `~/.local/bin/nuctl`.
4. **Deploy function** (lần đầu build image vài phút; weights không nằm trong image mà được mount):

   ```bash
   autolabel/deploy.sh
   ```

   Function đọc weights khi được gọi lần đầu, nên nếu đã deploy trước khi có weights thì chỉ cần tải weights về, **không cần deploy lại**.

   **Windows (Docker Desktop + GPU):** `powershell -ExecutionPolicy Bypass -File autolabel\deploy.ps1 gpu` (nuctl chạy trong container, không cài gì lên máy). Bản GPU cài torch cu128 từ các wheel Linux/py3.12 đặt sẵn trong `autolabel/wheels/` (git-ignore, ~4 GB: torch 2.10.0, torchvision 0.25.0, các gói `nvidia-*`, triton...); tải trước (ví dụ `curl -C -` để nối tiếp khi mạng rớt) rồi mới deploy; build cài `--no-index` nên thiếu wheel là báo lỗi ngay.

## Dùng trong CVAT

1. Mở task → **Actions → Automatic annotation** → chọn **SAM 3 auto-label (Day03)**. Label tự ghép theo tên. Có thể tick "Clean old annotations".
2. Chờ: CPU mất khoảng vài chục giây đến vài phút mỗi ảnh (lần đầu thêm thời gian nạp model). Xem log: `docker logs -f nuclio-nuclio-sam3-autolabel`.
3. Mở job, **sửa** mask (class, biên, object thiếu/thừa), Save, export như README chính rồi đặt ZIP vào `submissions/<task>.zip`.
4. Đo phần do tool làm:

   ```bash
   pip install -r requirements.txt     # numpy, pillow, pycocotools
   python3 autolabel/compare.py
   ```

   - Object (instance, thing của panoptic): object cuối được tính "từ tool" nếu có mask tool cùng class, IoU ≥ 0.5 (ghép 1-1). Bảng có số object tự vẽ thêm và số object tool bị xoá.
   - Pixel (semantic, stuff của panoptic): % pixel có nhãn trong bài cuối giữ đúng class tool gán.
   - Panoptic: % task = trung bình % object và % pixel. Dòng tổng = trung bình có trọng số theo điểm task.
5. Khi coach phát ground truth: giải nén ZIP reference **ra ngoài repo** (ví dụ `~/day3-ref`, sẽ có `~/day3-ref/data/tiers/...`) rồi

   ```bash
   python3 autolabel/compare.py --gt ~/day3-ref
   ```

   Cột thêm "Mask gốc vs GT" chấm mask gốc của tool bằng metric của `lab_utils.py` (mIoU / matched IoU×recall / PQ). Không copy ground truth vào repo.

## Chỉnh kết quả

- Sửa `config.json` (prompt, ngưỡng, `min_area`, `nms_iou`) rồi chạy lại Automatic annotation, không cần deploy lại. Ví dụ class bị bỏ sót → hạ ngưỡng class đó (`"thresholds": {"sidewalk": 0.25}`); class nhận nhầm → nâng ngưỡng; thêm prompt đồng nghĩa (`"vegetation": ["vegetation", "tree", "grass"]`).
- Sửa code trong `nuclio/` thì phải chạy lại `autolabel/deploy.sh`.
- Thử nhanh 1 ảnh không qua CVAT:

  ```bash
  PORT=$(docker port nuclio-nuclio-sam3-autolabel 8080 | head -1 | cut -d: -f2)
  python3 -c "import base64,json;print(json.dumps({'image':base64.b64encode(open('data/checkpoints/cp4_curb/images/7d83710e-4697c3b2.jpg','rb').read()).decode()}))" \
    | curl -s -X POST -H 'Content-Type: application/json' --data @- "http://localhost:$PORT" | head -c 300
  ```

## Test (không cần weights)

```bash
python3 -m unittest discover -s autolabel/tests -v
```

## Lưu ý

- Không commit `autolabel/weights/` (đã git-ignore), token HF hay ground truth.
- `autolabel/raw/` mặc định commit được (bằng chứng, dung lượng nhỏ).
- Chưa có weights thì Automatic annotation báo lỗi; log function ghi rõ `SAM 3 weights not found at autolabel/weights/sam3.pt`.
- Dùng checkpoint `facebook/sam3` (`sam3.pt`). `facebook/sam3.1` là bản multiplex cho video, chưa kiểm tra với image model nên không dùng.
- SAM 3 chính thức chỉ hỗ trợ CUDA; image build vá 2 chỗ `device="cuda"` cố định trong `sam3/model/position_encoding.py` và `decoder.py` thành `cpu` (xem `nuclio/Dockerfile`).
