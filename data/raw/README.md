# data/raw

Dữ liệu gốc, **không commit** (chỉ file này và `.gitkeep`). Mỗi bộ dữ liệu một thư mục:

```
data/raw/
├── mekong/                       # bộ chính: bản đồ lũ MODIS ĐBSCL
│   ├── kpftzmsyyz-2.zip          # tải từ Mendeley Data, DOI 10.17632/kpftzmsyyz.2
│   └── kpftzmsyyz-2/             # giải nén từ zip (xem bên dưới)
│       ├── Flooding/             # 565 bản đồ lũ (485 .tif GeoTIFF + 80 .tiff ENVI raw kèm .hdr)
│       └── Land use/             # 21 bản đồ sử dụng đất
├── mekong_hydrology/
│   └── fm996rrfyz-3.zip          # số liệu thủy văn Mê Kông THEO NĂM (xem bên dưới)
└── haiphong_tide/
    └── Data_2021-2022.xlsx       # mực nước theo giờ, Hải Phòng, 2021–2022
```

Giải nén bản đồ lũ: `python -m zipfile -e data/raw/mekong/kpftzmsyyz-2.zip data/raw/mekong/`

## Các bộ phụ

| Thư mục | Nội dung | Dùng để |
|---|---|---|
| `mekong_hydrology/` | Sheet `WL_TanChau` (mực nước năm, 1980–2016), `WL_MyThuan` (1980–2015), độ mặn tối đa theo năm, mưa năm của thượng/hạ lưu, lưu lượng trung bình theo thời kỳ | **Kiểm chứng** chuỗi diện tích ngập (cấp năm) và bảng Level 3. Không dùng làm biến dự báo 8 ngày vì chỉ có số liệu theo năm |
| `haiphong_tide/` | Mực nước theo giờ tại Hòn Dấu và Trung Trang (sông Văn Úc), lưu lượng giờ 2021 | Vùng khác (miền Bắc), chỉ 2 năm, chịu ảnh hưởng thủy triều. **Không** ghép với chuỗi ĐBSCL. Chỉ để tham khảo hoặc Future Work |

Bộ `mekong_hydrology` đi kèm bài báo của nguồn dữ liệu riêng, cần ghi trích dẫn đầy đủ trước khi đưa vào báo cáo.
