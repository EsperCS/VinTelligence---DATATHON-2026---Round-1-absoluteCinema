# Submission của team absoluteCinema cho cuộc thi DATATHON2026

Repository này hiện có hai phần chính:

- `Task1`: notebook phân tích dữ liệu và trả lời câu hỏi trắc nghiệm.
- `Task3`: pipeline tái tạo file dự báo `submission_sniper_3.csv`.

## Cấu trúc thư mục

```text
.
├── README.md
├── Task1
│   ├── answers.ipynb
│   ├── dap_an.txt
│   └── dataset
│       └── round1
└── Task3
    ├── requirements.txt
    ├── data
    └── src
```

## Task1

`Task1/answers.ipynb` là notebook dùng để đọc dữ liệu trong `Task1/dataset/round1`, tính toán cho từng câu hỏi, và suy ra đáp án.

### Dữ liệu đầu vào

Notebook đang được cấu hình để dùng dataset tại:

```text
Task1/dataset/round1
```

Trong đó có các file như:

- `orders.csv`
- `order_items.csv`
- `products.csv`
- `returns.csv`
- `web_traffic.csv`
- `customers.csv`
- `geography.csv`
- `payments.csv`
- `sales.csv`

### Cách chạy Task1

1. Cài Python 3.10+.
2. Cài các thư viện cần thiết:

```bash
pip install pandas jupyter
```

3. Mở Jupyter Notebook:

```bash
jupyter notebook
```

4. Mở file:

```text
Task1/answers.ipynb
```

5. Chạy lần lượt các cell trong notebook để sinh kết quả cho từng câu.

### Kết quả Task1

- File notebook: `Task1/answers.ipynb`
- File đáp án chữ cái: `Task1/dap_an.txt`

## Task3

`Task3` chứa pipeline để tái tạo submission cuối cùng:

```text
Task3/data/submission_sniper_3.csv
```

### Môi trường

- Python `3.10+`

### Cài thư viện

Di chuyển vào thư mục `Task3` rồi cài dependency:

```bash
cd Task3
pip install -r requirements.txt
```

### Dữ liệu đầu vào

Các file đầu vào của `Task3` nằm trong:

```text
Task3/data
```

Ví dụ:

- `sales.csv`
- `orders.csv`
- `order_items.csv`
- `products.csv`
- `promotions.csv`
- `inventory.csv`
- `web_traffic.csv`
- `daily_feature_table.csv`
- `sample_submission.csv`
- `future_promo_calendar_features.csv`

### Chạy pipeline Task3

Từ thư mục `Task3`, chạy:

```bash
python src/reproduce_submission_sniper_3.py --mode full
```

Kết quả cuối cùng:

```text
Task3/data/submission_sniper_3.csv
```

Nếu muốn chỉ dựng output cuối từ các artifact đã có sẵn:

```bash
python src/reproduce_submission_sniper_3.py --mode fast
```

Nếu muốn cho phép ghi đè các file đã sinh:

```bash
python src/reproduce_submission_sniper_3.py --mode full --force
```

### Ghi chú về Task3

- Entry-point chính: `Task3/src/reproduce_submission_sniper_3.py`
- Script này kiểm tra schema đầu ra gồm các cột: `Date`, `Revenue`, `COGS`
- Pipeline có thể chạy ở `fast` hoặc `full`

## Gợi ý sử dụng nhanh

- Nếu bạn cần xem và chạy phần trả lời câu hỏi dữ liệu: vào `Task1/answers.ipynb`
- Nếu bạn cần tái tạo submission dự báo: vào `Task3` và chạy `src/reproduce_submission_sniper_3.py`
