# IM-DD Optical Access Link — Trade-off Study

Dự án này mô phỏng một tuyến thông tin quang IM-DD (Intensity Modulation / Direct Detection) sử dụng thư viện OptiCommPy.

Mục tiêu của dự án là so sánh hiệu năng của OOK và 4-PAM trong các điều kiện khác nhau của bộ thu và kênh truyền quang.

## Nội dung mô phỏng

- Mô phỏng tuyến IM-DD cơ bản.
- Quan sát Eye Diagram tại phía phát và phía thu.
- Khảo sát BER theo công suất quang thu.
- Khảo sát BER theo băng thông bộ thu.
- Khảo sát BER theo chiều dài sợi quang.
- Khảo sát BER theo hệ số tán sắc.
- So sánh hiệu năng giữa OOK và 4-PAM (TODO)

## Files

- `run_all_experiments.py` — main script, chạy toàn bộ nội dung mô phỏng và lưu hình.
- `imdd_lib.py` — thư viện mô phỏng, bao gồm các khối của hệ thống IM-DD, các hàm khảo sát và vẽ kết quả.
- `requirements.txt` — danh sách các thư viện Python cần cài đặt để chạy dự án.

## Cài đặt

```bash
pip install -r requirements.txt
pip install OptiCommPy      # thư viện core
```

## Chạy

```bash
# Chạy đầy đủ, đúng tham số như notebook gốc (nBits=100,000 mỗi lần, có thể mất thời gian)
python run_all_experiments.py

# Chạy nhanh để kiểm tra pipeline hoạt động (nBits nhỏ, sweep thô — BER sẽ nhiễu, không dùng để lấy số liệu thật)
python run_all_experiments.py --quick

# Chỉ định thư mục lưu kết quả (mặc định: ./outputs)
python run_all_experiments.py --output-dir results

# Hiện hình trực tiếp khi chạy (ngoài việc lưu file)
python run_all_experiments.py --show

# Bỏ qua phần 3 (single-run sanity check + eye diagram) nếu chỉ cần sweep
python run_all_experiments.py --skip-sanity
```

## Kết quả

Script tạo 8 hình PNG trong thư mục output (mặc định `outputs/`):

| File | Nội dung |
|---|---|
| `03_eye_tx_ook.png`, `03_eye_rx_ook.png` | Eye diagram Tx/Rx — OOK |
| `03_eye_tx_pam4.png`, `03_eye_rx_pam4.png` | Eye diagram Tx/Rx — PAM4 |
| `04_ber_vs_power.png` | Sweep 1 — BER vs công suất thu |
| `05_ber_vs_bandwidth.png` | Sweep 2 — BER vs băng thông receiver |
| `06_ber_vs_length.png` | Sweep 3 — BER vs chiều dài sợi quang |
| `07_ber_vs_dispersion.png` | Sweep 4 — BER vs tán sắc |


