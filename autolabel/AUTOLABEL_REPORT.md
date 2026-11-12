# SAM 3 auto-label — phần do tool làm

| Task | Loại | Điểm | Object cuối | Từ tool | Tự vẽ thêm | Tool bị xoá | % object | % pixel | % task | Mask gốc vs GT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy_semantic | semantic | 20 | — | — | — | — | — | 100.0% | 100.0% | 0.814 |
| medium_instance | instance | 32 | 90 | 90 | 0 | 0 | 100.0% | — | 100.0% | 0.746 |
| hard_panoptic | panoptic | 30 | 52 | 51 | 1 | 0 | 98.1% | 100.0% | 99.0% | 0.475 |
| cp1_holes | instance | 3 | 6 | 6 | 0 | 0 | 100.0% | — | 100.0% | — |
| cp2_slice | instance | 3 | 17 | 17 | 0 | 0 | 100.0% | — | 100.0% | — |
| cp5_occlusion | instance | 3 | 54 | 54 | 0 | 0 | 100.0% | — | 100.0% | — |
| cp3_thin | semantic | 3 | — | — | — | — | — | 100.0% | 100.0% | — |
| cp4_curb | semantic | 3 | — | — | — | — | — | 100.0% | 100.0% | — |
| cp6_coverage | semantic | 3 | — | — | — | — | — | 100.0% | 100.0% | — |

**Tổng (trung bình có trọng số theo điểm task, 9 task): 99.7%**
