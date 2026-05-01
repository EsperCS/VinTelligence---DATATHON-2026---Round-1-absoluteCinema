# README - Tái tạo `submission_sniper_3.csv`

## 1. Mục đích

Repository này được nộp kèm để ban tổ chức có thể kiểm tra và tái tạo file
submission cuối cùng:

```text
data/submission_sniper_3.csv
```

Submission này là kết quả của một pipeline dự báo nhiều tầng, trong đó dự báo
cuối được tạo bằng một bước ensemble bảo thủ trên hai tín hiệu trung gian:

```text
Revenue = 0.95 * current_best + 0.05 * stock_conservative
COGS    = Revenue * 0.8900
```

Trong đó:

- `current_best`: forecast anchor chính
- `stock_conservative`: lớp hiệu chỉnh theo ràng buộc tồn kho

Hai artifact trung gian trên **không cần được chuẩn bị thủ công**. Chúng được
tạo tự động trong quy trình full reproduction từ raw data.

---

## 2. Cấu trúc repository

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── reproduce_submission_sniper_3.py
│   ├── train_final_sniper_grid.py
│   ├── final_feature_prune_and_retrain.py
│   ├── train_spike_aware_model.py
│   ├── train_promo_regime_model.py
│   ├── train_spike_probability_gate.py
│   ├── adaptive_scaling_layer.py
│   ├── train_meta_scaling.py
│   ├── train_direct_seasonal_residual_model.py
│   ├── final_micro_calibration.py
│   └── train_stock_aware_scaling.py
└── data/
    ├── sales.csv
    ├── orders.csv
    ├── order_items.csv
    ├── products.csv
    ├── promotions.csv
    ├── inventory.csv
    ├── web_traffic.csv
    ├── daily_feature_table.csv
    ├── sample_submission.csv
    └── future_promo_calendar_features.csv
```

---

## 3. Yêu cầu môi trường

### Python

- Python `3.10+`

### Cài đặt thư viện

Sử dụng:

```bash
pip install -r requirements.txt
```

hoặc cài trực tiếp:

```bash
pip install numpy pandas scikit-learn lightgbm
```

---

## 4. Dữ liệu đầu vào

Các file đầu vào cần được đặt trong thư mục `data/`:

- `sales.csv`
- `orders.csv`
- `order_items.csv`
- `products.csv`
- `promotions.csv`
- `inventory.csv`
- `web_traffic.csv`
- `daily_feature_table.csv`
- `sample_submission.csv`
- `future_promo_calendar_features.csv` (nếu được cung cấp trong bộ dữ liệu cuộc thi)

---

## 5. Lệnh tái tạo chính

Lệnh chính để tái tạo submission cuối cùng:

```bash
python src/reproduce_submission_sniper_3.py --mode full
```

Đầu ra cuối cùng:

```text
data/submission_sniper_3.csv
```

Nếu workspace đã tồn tại các output generated từ lần chạy trước, có thể sử dụng:

```bash
python src/reproduce_submission_sniper_3.py --mode full --force
```

để ghi đè các output được tạo trong quá trình reproduce.

---

## 6. Giải thích pipeline

Script entry-point:

```text
src/reproduce_submission_sniper_3.py
```

Script này điều phối toàn bộ pipeline và tái tạo submission cuối từ raw data.

Trình tự các bước chính:

1. Huấn luyện và sinh `submission_pruned_ensemble.csv`
2. Huấn luyện và sinh `submission_spike_aware.csv`
3. Huấn luyện và sinh `submission_promo_regime.csv`
4. Tạo `submission_regime_ultra_15.csv`
5. Huấn luyện spike probability gate
6. Chạy adaptive scaling layer
7. Chạy meta scaling
8. Huấn luyện direct seasonal residual model
9. Chạy final micro calibration
10. Dựng `submission_cogs_ratio_8900.csv`
11. Dựng `submission_blend_direct_15_cogs8900.csv`
12. Huấn luyện stock-aware scaling
13. Sinh `submission_sniper_3.csv`

Giải thích:

- `submission_blend_direct_15_cogs8900.csv` là forecast anchor chính của nhánh cuối
- `submission_stock_scale_conservative.csv` là nhánh hiệu chỉnh bảo thủ theo tín hiệu tồn kho
- `submission_sniper_3.csv` là ensemble rất nhỏ giữa hai nhánh trên

Do đó, việc công thức cuối tham chiếu tới các artifact trung gian là một phần tự
nhiên của pipeline. Các artifact này được sinh ra trong quá trình chạy full
reproduction, không phải đầu vào thủ công.

---

## 7. Kiểm tra đầu ra

Trong quá trình tái tạo, script kiểm tra các điều kiện sau cho submission cuối:

- số dòng khớp với `sample_submission.csv`
- đúng thứ tự `Date`
- đúng schema:

```text
Date, Revenue, COGS
```

- không có missing values
- không có giá trị âm

---

## 8. File submission cuối cùng

File dùng để nộp leaderboard:

```text
data/submission_sniper_3.csv
```

---

## 9. Ghi chú cho ban tổ chức

Repository này được tổ chức theo hướng:

- full reproducibility từ raw data
- có `requirements.txt` để cài đặt môi trường
- có `README.md` mô tả cấu trúc thư mục, dữ liệu đầu vào và cách chạy lại
- có script entry-point rõ ràng:

```bash
python src/reproduce_submission_sniper_3.py --mode full
```

Nếu cần kiểm tra nhanh công thức cuối cùng, có thể xem thêm:

```text
src/train_final_sniper_grid.py
```

Script này mô tả trực tiếp các cấu hình sniper-grid và ứng viên
`submission_sniper_3.csv` tương ứng với công thức:

```text
Revenue = (0.950 * current_best + 0.050 * stock_conservative) * 1.0000
```
