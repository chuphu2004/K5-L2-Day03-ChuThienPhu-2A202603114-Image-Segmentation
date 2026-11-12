# Thiết kế: SAM 3 cho CVAT online qua REST API + PAT

Ngày: 2026-11-12 · Trạng thái: đã duyệt thiết kế, chờ duyệt spec. Làm **sau** khi Day03 local (nuclio GPU) chạy xong.

Liên quan: [thiết kế nuclio cho CVAT local](2026-11-11-sam3-autolabel-design.md) · [`autolabel/README.md`](../../../autolabel/README.md).

## Mục đích

Chương trình có một CVAT online (server riêng, không phải app.cvat.ai, chưa rõ bản Community hay Enterprise). Học viên **không được import/export** dữ liệu trên trang đó và không có quyền deploy nuclio lên server. Cần một cách để SAM 3 (pretrained, đã có `autolabel/weights/sam3.pt`) tự vẽ mask vào task trên CVAT online.

## Mục tiêu

- Lệnh `autolabel/run.ps1 -Target online -Task <id>` chạy SAM 3 trên GPU máy học viên, đọc frame của task qua REST API, ghi mask thẳng vào task. Đăng nhập bằng **Personal Access Token (PAT)**.
- Chạy được với **task Day03** (tự biết class và mode từ `data/` theo tên task) và **task bất kỳ** (class lấy từ label của task, mode qua cờ).
- Chạy được với mọi bản CVAT có PAT (≥ 2.48), kể cả Community.
- Có lệnh kiểm tra chỉ đọc (`-Check`) để thử server online trước khi đụng dữ liệu.

## Không làm

- Không đo "% do tool làm" cho task online (`compare.py` giữ nguyên cho Day03 local).
- Không train/fine-tune, không chế độ interactor (click điểm).
- Không đổi function nuclio, `compare.py`, hay quy trình Day03 local.
- Phương án B (AI agent) chỉ làm nếu server hỗ trợ, xem mục riêng.

## Phản mục tiêu

- PAT nằm trong git, image Docker, log, hay file raw.
- Xoá hoặc nhân đôi annotation người dùng đã vẽ mà không có lệnh rõ ràng.
- Dữ liệu (ảnh, mask) của task không phải Day03 bị commit lên GitHub.
- Chạy trên dữ liệu thật của chương trình khi chưa xác nhận quy định (mục "Điều kiện").

## Ràng buộc đã biết

- Nuclio chỉ dùng được khi điều khiển được server CVAT, nên không dùng được cho server của chương trình.
- AI agent (`cvat-cli function create-native` / `run-agent`) chỉ có trên CVAT Online và Enterprise.
- `cvat_sdk.auto_annotation.annotate_task(client, task_id, function, clear_existing=..., allow_unmatched_labels=...)` chạy trên mọi bản, chỉ với task ảnh (không video): SDK tải frame **gốc** (`quality="original"`), gọi `function.detect(context, image)` cho từng frame, map label theo **tên**, rồi upload **một lần ở cuối** (đã đọc source `cvat-sdk` v2.74.1 trong `cvat-day2/cvat-sdk`). Frame cache nằm trong `~/.cache/cvat-sdk` của container.
- Function cho SDK: `spec = DetectionFunctionSpec(labels=[label_spec(name, id, type="mask"), ...])`; `detect` trả `cvataa.mask(label_id, encode_mask(bool_mask, [x1, y1, x2, y2]))`; `context.frame_name` là tên file của frame.
- PAT: tạo ở **User settings → Security → Personal access tokens**, chọn read-only hoặc read/write, có hạn. SDK/CLI ≥ 2.48 đọc `CVAT_ACCESS_TOKEN`. Client đăng nhập bằng PAT không sửa được token/thông tin user.
- Máy chạy: Windows 11, RTX 3060 12 GB, Docker Desktop (WSL2), không cài tool lên host (chỉ Docker và venv `~/.venvs/day03`). Script chạy bằng PowerShell.

## Kiến trúc

```
Máy học viên (Docker Desktop, --gpus all)                 CVAT (local hoặc online)
┌──────────────────────────────────────────────┐  REST   ┌─────────────────────┐
│ container cvat.sam3.sdk                      │────────▶│ /api/tasks/<id>     │
│  cvat_run.py --task <id>                     │ frame ◀ │ frames              │
│   └ annotate_task(client, id, Sam3Function)  │ mask  ▶ │ annotations         │
│       └ sam3_func.py → nuclio/core.py        │         └─────────────────────┘
│ mount: autolabel/ → /autolabel (rw)          │
│        data/      → /data (ro)               │
│ env:   --env-file autolabel/.env.<tên>       │
└──────────────────────────────────────────────┘
```

## Thành phần (toàn bộ trong `autolabel/`)

| File | Việc |
| --- | --- |
| `nuclio/core.py` | Nhận thêm `load_processor` và `detect` chuyển từ `nuclio/main.py` (import `torch`/`sam3` bên trong hàm, nên test vẫn chạy không cần torch). `main.py` import lại từ `core`. Không đổi hành vi nuclio. |
| `sam3_func.py` (mới) | `Sam3Function(classes, mode, stuff, task_dir, cfg)`: `spec` = các class (`type="mask"`); `detect(context, image)` = SAM 3 → hậu xử lý theo mode bằng `core` → ghi raw → trả danh sách `cvataa.mask`. Thêm `create(**params)` để phương án B dùng lại. |
| `cvat_run.py` (mới) | CLI: `--task ID`, `--mode semantic\|instance\|panoptic`, `--stuff a,b`, `--append`, `--clear`, `--check`. Kết nối, đọc task, chọn class/mode, kiểm an toàn, gọi `annotate_task`, in tóm tắt. |
| `Dockerfile.sdk` (mới) | `FROM cvat.sam3.base.gpu` + `pip install cvat-sdk==${CVAT_SDK_VERSION}` (build arg, mặc định `2.74.1`). |
| `run.ps1` (mới) | `-Target local\|online` (chọn `autolabel/.env.<Target>`), `-SdkVersion` (mặc định `2.74.1`, image tag `cvat.sam3.sdk:<version>`), `-Test` (chạy test cần `cvat-sdk`), `-Task/-Check/-Mode/-Stuff/-Append/-Clear` (đổi thành cờ `--task/--check/...` của `cvat_run.py`). Build image nếu chưa có; `docker run --rm --gpus all --env-file ...`, mount như trên, vào mạng `cvat_cvat` nếu mạng đó tồn tại. |
| `.env.example` (mới) | Mẫu với giá trị giả: `CVAT_URL`, `CVAT_ACCESS_TOKEN`, `CVAT_ORG`. |
| `tests/test_core_sam3.py` (mới) | Test phần SAM 3 chuyển vào `core.py`, chạy trên host (venv). |
| `tests/test_sdk_func.py`, `tests/test_sdk_run.py` (mới) | Test `sam3_func.py` và `cvat_run.py`, cần `cvat-sdk` nên chạy trong container (`run.ps1 -Test`); trên host tự skip. |
| `README.md` | Thêm mục "Chạy với CVAT online (PAT)". |

`.gitignore` thêm: `autolabel/.env*`, `!autolabel/.env.example`, `autolabel/raw/_online/`.

## Luồng chạy

1. **Kết nối.** `make_client(CVAT_URL, access_token=CVAT_ACCESS_TOKEN)`; đặt organization nếu có `CVAT_ORG`. SDK tự so version server với version `cvat-sdk` khi kết nối và in cảnh báo nếu lệch major.minor (vẫn chạy); khi đó chạy lại với `-SdkVersion <version server>`.
2. **Đọc task.** Tên task, danh sách label, số frame, số shape/track đang có.
3. **Chọn class và mode.**
   - Tên task là khoá trong `data/manifest.json` (task Day03): mode và `stuff` lấy từ `classes.json`; class = `classes.json` ∩ label của task. Label lệch giữa hai bên → in cảnh báo. Raw ghi vào `raw/<tên task>/`.
   - Task khác: class = toàn bộ label của task; mode = `--mode` (mặc định `instance`); `panoptic` bắt buộc có `--stuff`. Raw ghi vào `raw/_online/<task id>/` (git-ignore).
   - `--mode`/`--stuff` luôn ghi đè giá trị tự suy ra.
4. **`--check` dừng ở đây**: in URL, version server, user, org, `GET /api/functions` trả gì (có dùng được agent không), tên task, label, số frame, số annotation đang có, class và mode sẽ dùng. Không tải frame, không ghi gì.
5. **Kiểm an toàn.** Task đã có shape/track mà không có `--append` hay `--clear` → dừng, in số lượng và cách chạy lại. `--clear` → in "sẽ xoá N shape, M track" rồi mới chạy.
6. **Chạy.** `annotate_task(client, id, Sam3Function(...), clear_existing=args.clear, allow_unmatched_label=True)`. Mỗi frame: `set_image` một lần → mỗi class chạy các prompt trong `config.json` (không có thì prompt = tên class) với ngưỡng theo class → hậu xử lý theo mode → ghi raw `<thư mục raw>/<tên frame không đuôi>.json` (cùng format bản nuclio) → trả mask.
7. **Tóm tắt.** Số frame, số object theo class, tổng giây, đường dẫn raw.

Model nạp một lần mỗi lần chạy lệnh. Ngưỡng nhập trên CVAT không dùng; mọi chỉnh sửa nằm trong `config.json`, có hiệu lực ở lần chạy sau, không cần build lại.

## Lỗi

| Tình huống | Hành vi |
| --- | --- |
| 401 | "Token sai hoặc hết hạn", gợi ý tạo PAT mới. Không in token. |
| 403 / 404 task | "Không có quyền hoặc không thấy task <id>", gợi ý kiểm `CVAT_ORG`. |
| Không có class nào khớp | Dừng, in label của task và class mong đợi. |
| Không có `sam3.pt` | Dừng, chỉ tới `download_weights.sh`. |
| Video task | Dừng: SDK chỉ hỗ trợ task ảnh. |
| Không tới được `CVAT_URL` | "Cannot reach CVAT", gợi ý kiểm URL/VPN. |
| SAM 3 lỗi giữa chừng | Thoát mã khác 0. SDK chỉ upload ở cuối nên task không đổi. |
| Mạng rớt lúc upload | Chạy `--check` xem số shape; nếu đã ghi một phần thì chạy lại với `--clear` hoặc xoá tay. |

## Bảo mật và dữ liệu

- PAT chỉ đi vào container qua `--env-file`; không nằm trong image, git, log, raw. File `.env.local` / `.env.online` do học viên tự gõ, không dán vào chat.
- PAT online: quyền read/write (tool phải ghi annotation), hạn khoảng 3 ngày, thu hồi khi xong.
- Raw của task không phải Day03 nằm trong `raw/_online/` (git-ignore) để dữ liệu của chương trình không lên GitHub.
- Không tool nào tự xoá annotation nếu không có `--clear`.
- Frame tải về chỉ nằm trong cache của container, mất khi container dừng (`--rm`); không ghi ảnh ra ổ host.

## Test

1. **Unit** (không GPU, không server; `tests/test_core_sam3.py` trên host, `tests/test_sdk_*.py` trong container): processor giả trả mask cố định → `spec` đúng tên/`type="mask"`; `detect` trả mask đúng label id và bbox theo từng mode; `encode_mask` giải ngược khớp mask gốc; chọn class/mode cho task Day03, task khác, task panoptic thiếu `--stuff`; kiểm an toàn với task đã có annotation. 17 test cũ vẫn pass.
2. **Model**: trong container `cvat.sam3.sdk`, gọi `Sam3Function.detect` trên 1 ảnh `data/` (không CVAT); `nvidia-smi` thấy GPU.
3. **Diễn tập Day03 trên CVAT local** (v2.74.1, có PAT): tạo PAT local ghi vào `.env.local` (`CVAT_URL=http://cvat_server:8080`, container vào mạng `cvat_cvat`); tạo **task test riêng** trùng tên Day03 (ví dụ `cp2_slice`, 1 ảnh, label từ `cvat-labels.json`), không đụng 9 task bài làm. Chạy `--check`, rồi chạy thật; mở UI thấy mask đúng label; số object theo class trong raw bằng số của bản nuclio cho cùng ảnh (lệch → kiểm frame nén hoặc chọn class/mode sai). Chạy lại lần hai → tool dừng vì task đã có annotation; thử `--append` và `--clear`.
4. **`--check` trên CVAT online**: chỉ đọc. Xác nhận PAT, quyền, org, version, có/không agent.
5. **Chạy thật trên online**: 1 task nhỏ được chương trình cho phép, chỉ sau mục "Điều kiện". Xem kết quả trong UI rồi mới chạy task khác.

## Phương án B: AI agent (chỉ khi tầng 4 thấy `/api/functions`)

`cvat-cli function create-native "SAM 3" --function-file autolabel/sam3_func.py -p ...` một lần, rồi `cvat-cli function run-agent <id> ...` chạy trong container, phải chạy suốt lúc có người dùng. Agent không biết task nào gọi, nên class và mode cố định qua `-p` lúc đăng ký. Function sẽ hiện cho mọi thành viên organization. Nếu `/api/functions` trả 404 thì bỏ phương án này.

## Điều kiện trước khi chạy dữ liệu thật trên online

- Chương trình xác nhận được phép tải ảnh của task về máy học viên qua API để chạy model. Nếu không, chỉ còn cách nhờ admin deploy bản nuclio lên server (phương án C, ngoài phạm vi spec này).
- Tầng 3 và 4 đạt.

## Đã kiểm trong source CVAT v2.74.1

- `annotate_task(..., clear_existing, allow_unmatched_labels, conf_threshold, conv_mask_to_poly)`; upload một lần ở cuối (`update_annotations` khi clear, `partial_update_annotations("create")` khi không).
- `make_client(host, access_token=...)` (PAT); `client.organization_slug = "<org>"`; SDK tự cảnh báo khi lệch version (`allow_unsupported_server=True` mặc định).
- `cvat_sdk.masks.encode_mask(bool HxW)` tự tính bbox; `decode_mask(points, image_width=, image_height=)`.
- `cvat_server` có `ALLOWED_HOSTS: '*'`, nghe cổng 8080 trong mạng `cvat_cvat`, nên `http://cvat_server:8080` gọi thẳng được (không qua traefik).
- `cvat-sdk` chỉ cần numpy cho `masks` (extra `masks` đòi numpy ≥ 2, nhưng code chạy với numpy 1.26 của image nền); cài `cvat-sdk` **không kèm extra** để giữ numpy của `sam3`.

## Còn phải kiểm lúc implement

- `cvat-sdk==2.74.1` có trên PyPI (nếu không: bản 2.74.x gần nhất).
- `encode_mask`/`decode_mask` chạy đúng với numpy 1.26 (test tầng 1 bắt).

## Thứ tự làm

Chỉ bắt đầu khi Day03 local (image `cvat.sam3.base.gpu`, function nuclio) chạy xong, vì dùng chung image và `nuclio/main.py`.
