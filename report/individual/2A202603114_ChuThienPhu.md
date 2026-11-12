# Báo cáo thực hành Day 3 — điền trực tiếp trong fork của bạn

**Cách dùng:** Thay mọi dấu `…` bằng bài làm thật của bạn trước khi nộp link fork trên [LMS](https://aithucchien.vinuni.edu.vn). Giữ nguyên bốn mục và bảng để coach đọc nhanh. Viết ngắn, cụ thể theo ảnh/vùng; không cần thuật ngữ chuyên sâu. Ví dụ trong [hướng dẫn mẫu](../../reports/REPORT_TEMPLATE.md) chỉ giúp hiểu cách điền, không phải câu trả lời để chép lại.

- Mã học viên theo lớp: 2A202603114
- Ngày / CVAT local: lab 11/11/2026, hoàn thiện 12/11/2026 · CVAT local v2.74.1 (Docker Desktop, Windows 11) bật serverless (nuclio 1.16.3)
- Công cụ đã dùng: tool **SAM 3 auto-label** tự đóng gói trong [`autolabel/`](../../autolabel/README.md) (function nuclio chạy SAM 3 pretrained `facebook/sam3` bằng text prompt trên GPU, gọi qua **Actions → Automatic annotation**) để tạo mask cho cả 9 task, sau đó xem lại trong CVAT rồi Save/export. Trợ lý AI (Claude Code) hỗ trợ viết và đóng gói code của tool.

Mã học viên là mã lớp cấp; không cần ghi họ tên trong report nếu kênh [LMS](https://aithucchien.vinuni.edu.vn) đã nhận diện bạn. Chỉ ghi công cụ thật sự đã dùng; không có SAM vẫn làm bài bình thường.

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | `easy_semantic.zip` | 3 / 3 | 20 |
| medium_instance | `medium_instance.zip` | 3 / 3 | 32 |
| hard_panoptic | `hard_panoptic.zip` | 2 / 2 | 30 |
| cp1_holes | `cp1_holes.zip` | 1 / 1 | 3 |
| cp2_slice | `cp2_slice.zip` | 1 / 1 | 3 |
| cp5_occlusion | `cp5_occlusion.zip` | 1 / 1 | 3 |
| cp3_thin | `cp3_thin.zip` | 1 / 1 | 3 |
| cp4_curb | `cp4_curb.zip` | 1 / 1 | 3 |
| cp6_coverage | `cp6_coverage.zip` | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: 000000373353.jpg (vẽ tay trên một máy CVAT khác trước khi dùng tool; bản vẽ tay đó không nằm trong ZIP nộp)
- Class và quy tắc tôi dùng để chọn biên: person, car, bus/ vẽ sát phần nhìn thấy, những phần của 1 thực thể nhưng bị che khuất sẽ được gán của cùng 1 instance, các xe khác nhau đứng che khuất nhau được để là các instance khác nhau
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do: gợi ý đúng, không sửa gì và chọn sử dụng phần thực hiện của tool vì ôm sát tốt hơn rất nhiều so với vẽ tay.

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: hard_panoptic/ 000000460147.jpg/ vùng car nằm trên truck 
- Lỗi thuộc loại: sai lớp 
- Bằng chứng tôi nhìn thấy: tool gán nhãn là building, do độ tối và hình dạng của chiếc xe khá vuông vức
- Quy tắc và hành động sửa: đè lên bằng 1 layer car
- Sau sửa đã Save và export lại chưa? sửa trước khi export

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): tự chạy `scoring/score.py` với reference (sau khi reference được phát): `hard_panoptic` PQ 0.475 (mask gốc của tool) → 0.476 (bài nộp sau sửa), 18.4/30. Scorecard ba tier tối đa **82**, không phải điểm cuối trên 100. Không tự ghi PASS/top 3/bonus; người phụ trách xác nhận theo tiêu chí lớp. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| `cp4_curb/7d83710e-4697c3b2.jpg`, vùng lát xi măng bên phải, chỗ xe SUV trắng đỗ (lối xe vào) | `sidewalk`, hoặc `road` vì xe chạy và đỗ trên đó | Ranh road–sidewalk theo chức năng và bó vỉa, không theo màu. Vùng này nằm sau bó vỉa, cùng mặt lát với vỉa hè. | Giữ `sidewalk`. Hỏi coach: lối xe vào băng qua vỉa hè có tính là sidewalk không? |
| `medium_instance/000000373353.jpg`, người và xe rất nhỏ ở cuối đường | Vẽ từng người/xe thành instance riêng, hoặc bỏ qua vì quá nhỏ và không thấy rõ biên | Quy tắc: mỗi vật nhìn thấy là một instance. Nhưng ở xa, nhiều người dính nhau, khó tách biên. | Giữ các object nhìn rõ. Hỏi coach: có ngưỡng kích thước tối thiểu để bỏ object quá nhỏ không? |
| `hard_panoptic/000000460147.jpg`, mặt đường màu đỏ ở giữa ảnh | `road` vì xe đang chạy trên đó, hoặc `sidewalk` vì là mặt lát màu khác | Quy tắc theo chức năng: xe chạy nên là road. Sau khi xem reference: reference gán phần lớn vùng này là `sidewalk`. | Bài nộp để `road`. Hỏi coach: mặt đường màu đỏ này theo quy ước là gì? |

## Phụ lục: tool SAM 3 auto-label

Mô tả, cách chạy và code: [`autolabel/README.md`](../../autolabel/README.md). Mask gốc tool sinh ra cho từng ảnh lưu ở `autolabel/raw/<task>/`.

Điểm của **mask gốc của tool** (chưa sửa tay) so với reference ba tier, đo bằng metric của `lab_utils.py`:

| Tier | Metric | Config mặc định (trước khi xem reference) | Config cuối (chỉnh **sau khi** xem reference) |
| --- | --- | ---: | ---: |
| easy_semantic | mIoU | 0.791 | 0.814 |
| medium_instance | mean matched IoU × recall | 0.738 | 0.746 |
| hard_panoptic | PQ | 0.427 | 0.475 |

`autolabel/config.json` (ngưỡng theo class, prompt, `min_area`) và bước lọc trùng giữa các class trong `autolabel/nuclio/core.py` được chỉnh **sau khi reference được phát**, nên cột "Config cuối" là kết quả đã tune trên chính bộ ảnh chấm. Với `hard_panoptic`, ảnh `000000460147` có phần lớn mặt đường đỏ ở giữa được reference gán `sidewalk` trong khi tool gán `road`; tôi không ép tool theo riêng ảnh này.
