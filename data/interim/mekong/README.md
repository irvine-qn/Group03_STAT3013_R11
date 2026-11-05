# data/interim/mekong

Chuỗi trung gian dựng từ ảnh gốc. CSV nhỏ nên được commit.

| File | Nguồn | Ghi chú |
|---|---|---|
| `series_mean.csv` | `floodbench.data.build_series` (quy lưới về 500 m bằng **mean**) | Bản đang dùng. 565 hàng, có cột `file_md5` để phát hiện ảnh trùng |
| `legacy/series_nearest_v1.csv` | Script cũ, quy lưới bằng **nearest-neighbor** | Bản cũ, giữ lại để so sánh "cũ/mới" (Plans M1-04). Không dùng cho phân tích |
| `legacy/zones_nearest_v1.csv` | Script chia vùng cũ, cũng dùng nearest | Có cột `z0/z1/z2_*` (vùng proxy theo tọa độ). Cần dựng lại từ lưới đã sửa |
| `legacy/features_v4.csv` | Bảng đặc trưng (trễ, mục tiêu 8/16 ngày) do thành viên nhóm tạo từ `series_mean.csv` | Chưa rõ script tạo. Chỉ để tham khảo |

Bảng đã làm sạch, sẵn sàng cho mô hình nằm ở `data/processed/mekong/`.
