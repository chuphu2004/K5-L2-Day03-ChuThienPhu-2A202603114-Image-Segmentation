# Thiết kế: SAM 3 auto-label cho CVAT (Day03 bonus)

Ngày: 2026-11-11 · Trạng thái: đã duyệt để làm (học viên cho làm luôn, weights SAM 3 đang chờ Meta duyệt).

## Mục đích

Giảng viên cộng điểm nếu học viên **tự đóng gói** một tool auto-segmentation và tool tự label được **50%+**. Học viên hiểu 50%+ theo cả hai nghĩa và muốn càng cao càng tốt:

1. Hơn nửa số mask/pixel trong bài nộp cuối có nguồn gốc từ tool (học viên chỉ sửa).
2. Mask gốc của tool, khi chấm với ground truth (phát cuối buổi), đạt điểm cao.

Model được chỉ định: **SAM 3** (`facebookresearch/sam3`, checkpoint gated `facebook/sam3` hoặc `facebook/sam3.1` trên Hugging Face). Được dùng pretrained, miễn là tự đóng gói.

## Mục tiêu

- Một function **nuclio** cắm vào CVAT local đang chạy (v2.74.1, checkout tại `/home/phu/Documents/AI20K/cvat-day2`). Trong CVAT, học viên bấm **Actions → Automatic annotation** trên task, chọn function, mask hiện ra với label trùng tên `classes.json`.
- Tool **tự lưu mask gốc** mỗi lần được gọi.
- Một lệnh **so sánh** mask gốc với ZIP đã sửa trong `submissions/`, in bảng "% do tool làm" cho từng task và tổng có trọng số. Nếu có ground truth, chấm thêm mask gốc bằng metric của `lab_utils.py`.
- Chạy được trên **CPU** (máy không có GPU; 12 CPU, 15 GB RAM).

## Không làm

- Không train/fine-tune.
- Không tự tạo hay sửa ZIP nộp bài. ZIP nộp vẫn export từ CVAT như README.
- Không chế độ interactor (click điểm). Có thể làm sau nếu còn thời gian.
- Không đóng gói kiểu Qualcomm/ONNX.

## Phản mục tiêu (làm thế này là hỏng dù chạy được)

- Commit HF token, weights, hoặc ground truth vào git. RULES.md cấm token; repo test cấm thư mục `groundtruth`.
- Làm vỡ ràng buộc của repo: sửa `data/`, `submissions/` (ngoài ZIP task), `notebooks/` (phải đúng 1 `.ipynb`), `tests/`, workflow. Toàn bộ code mới nằm trong `autolabel/` ở gốc fork.
- Làm mất annotation đang có trong CVAT khi bật serverless (volume Docker phải giữ nguyên).
- Số liệu "% do tool" bị thổi phồng: phải đếm trung thực, mask bị sửa nhiều thì không tính là của tool.

## Ràng buộc đã biết

- `facebookresearch/sam3`: Python ≥3.12, PyTorch ≥2.7; README chỉ hỗ trợ chính thức CUDA. `build_sam3_image_model` tự chọn `cpu` khi không có CUDA; `Sam3Processor(model, device=...)` mặc định `"cuda"` nên phải truyền `"cpu"`. Có thể còn chỗ gọi CUDA cứng (ví dụ autocast). Builder kiểm và vá bằng cách nhỏ nhất (monkeypatch/biến môi trường), ghi `shortcut:` nếu vá tạm.
- API ảnh: `processor.set_image(img)` một lần, rồi `processor.set_text_prompt(state, prompt="car")` cho từng class → `masks, boxes, scores`.
- Weights gated: tải trên **host** bằng `hf download` với token của học viên, lưu vào thư mục bị git-ignore (`autolabel/weights/`), **mount** vào container. Không truyền token vào build image.
- CVAT gửi cho detector chỉ `image` (base64) và `threshold`; **không gửi tên file hay label của task** (`cvat/apps/lambda_manager/views.py`). Function phải tự xác định ảnh.
- Nuclio: compose `components/serverless/docker-compose.serverless.yml` trong checkout CVAT (dashboard `quay.io/nuclio/dashboard:1.16.3-amd64`), deploy bằng `nuctl` cùng phiên bản. Mẫu tham khảo: `serverless/pytorch/facebookresearch/sam/nuclio/` và `serverless/deploy_cpu.sh`.
- CPU chậm: `eventTimeout` của function và timeout phía CVAT phải đủ dài (ước tính vài chục giây đến vài phút mỗi ảnh). Builder kiểm thực tế.

## Cách hoạt động (đã quyết)

**Xác định ảnh và task.** Function mount `data/` của fork (read-only). Khi nhận ảnh, so thumbnail nhỏ (ví dụ 32×32 xám) với 14 ảnh trong `data/tiers/*/images` và `data/checkpoints/*/images`, lấy ảnh gần nhất dưới một ngưỡng. Từ đó biết tên file, task, `classes.json` (class list, `type`, `stuff`). Không khớp ảnh nào → dùng hợp tất cả class và loại `instance`, vẫn lưu raw dưới `unknown/`.

**Label của function.** `function.yaml` khai báo hợp tất cả class của 9 task (14 tên: road, sidewalk, building, vegetation, sky, person, bicycle, car, motorcycle, bus, truck, traffic light, pole, traffic sign) để CVAT tự ghép theo tên. Function chỉ chạy prompt cho class của task đã nhận diện.

**Suy luận.** Một lần `set_image`, mỗi class một hoặc vài text prompt (bảng prompt có thể chỉnh, ví dụ `vegetation` → "vegetation", "tree"; gộp mask của các prompt cùng class). Ngưỡng tin cậy theo class có thể chỉnh trong config.

**Hậu xử lý theo loại task.**
- `instance`: mỗi mask SAM 3 là một object; bỏ mask quá nhỏ; NMS theo mask IoU trong cùng class.
- `semantic`: mỗi pixel thuộc tối đa một class (class có điểm cao nhất); mỗi class trả về một mask (có thể nhiều mảng rời).
- `panoptic`: stuff như semantic, thing như instance; thing đè lên stuff.

**Trả về CVAT.** Định dạng detector `type: "mask"` như các function mask có sẵn của CVAT (RLE + bbox). Builder đối chiếu với mã CVAT v2.74.1 để đúng format.

**Lưu mask gốc.** Mỗi lần gọi ghi vào `autolabel/raw/<task>/<tên_ảnh>.json` (mount từ host, git-ignore hay không do học viên quyết khi nộp; mặc định **commit được** vì là bằng chứng, dung lượng nhỏ): danh sách object `{label, score, rle, bbox}`, kèm thời gian và version model. Gọi lại thì ghi đè.

**So sánh** (`autolabel/compare.py`, CLI, chạy trên host, chỉ dùng thư viện đã có trong `requirements.txt` của repo):
- Đọc `autolabel/raw/` và ZIP trong `submissions/` (dùng loader trong `lab_utils.py`).
- Instance/panoptic thing: % object cuối khớp một mask tool (cùng class, IoU ≥ 0.5); số object tự vẽ thêm; số object tool bị xoá.
- Semantic/panoptic stuff: % pixel có nhãn trong bài cuối giữ đúng class tool gán.
- Bảng theo task + dòng tổng (trung bình có trọng số theo điểm task) → in ra và ghi `autolabel/AUTOLABEL_REPORT.md`.
- Tùy chọn `--gt <thư mục reference>`: chấm mask gốc bằng metric của `lab_utils.py` (mIoU / matched IoU×recall / PQ). Ground truth không bao giờ copy vào repo.

**Vận hành.** `autolabel/README.md` (tiếng Việt) hướng dẫn: xin quyền HF → `hf auth login` → `download_weights.sh` → bật serverless CVAT → `deploy.sh` → chạy auto annotation → sửa → export → `compare.py`.

## Để builder quyết

Cách bật serverless an toàn với stack đang chạy; cấu trúc file trong `autolabel/`; prompt và ngưỡng mặc định; cách vá CPU; ngưỡng khớp thumbnail; chọn `sam3` hay `sam3.1` (ưu tiên checkpoint nào có trước).

## Trạng thái khi bàn giao (vì weights chưa có)

Mọi thứ build và deploy được; phần thiếu duy nhất là file weights. Test không cần weights: khớp thumbnail, hậu xử lý semantic/instance/panoptic, RLE, `compare.py` trên dữ liệu tổng hợp. Khi có weights: tải về, redeploy, chạy thử 1 ảnh, chỉnh prompt/ngưỡng.
