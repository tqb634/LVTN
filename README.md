# IM-DD Optical Access Link — Trade-off Study

Dự án này mô phỏng một tuyến thông tin quang IM-DD (Intensity Modulation / Direct Detection) sử dụng thư viện **OptiCommPy**.

Mục tiêu của đề tài là khảo sát, so sánh và đánh giá định lượng hiệu năng truyền dẫn (BER, Q-factor, EVM) giữa hai định dạng điều chế **OOK** và **4-PAM** dưới tác động của công suất thu, giới hạn băng thông bộ thu, tán sắc sắc thể (Chromatic Dispersion) và chiều dài sợi quang.

## Ghi nhận

Đồ án này được phát triển dựa trên thư viện mô phỏng truyền thông quang OptiCommPy.

Một số cấu trúc mô phỏng và quy trình khởi tạo được tham khảo và phát triển từ tài liệu hướng dẫn chính thức cũng như các ví dụ minh họa (example notebooks) của OptiCommPy.

Tài liệu tham khảo:
- https://opticommpy.readthedocs.io/en/latest/getting_started.html
- https://github.com/edsonportosilva/OptiCommPy/blob/main/examples/basic_OOK_transmission.ipynb

## Nội dung mô phỏng

- **Mô phỏng toàn tuyến IM-DD**: Khối phát (OOK/4-PAM Mã hóa Gray + MZM) $\rightarrow$ Kênh sợi quang đơn mốt (Suy hao + Tán sắc) $\rightarrow$ Khối thu (Photodiode + Ước lượng ngưỡng phân định/pha lấy mẫu tối ưu từ Pilot).
- **Mắt tín hiệu (Eye Diagram)**: Trực quan hóa đồ thị mắt phía phát và phía thu.
- **Power Sweep**: Khảo sát BER theo công suất thu ($P_{rx}$) tại các mức tốc độ bit (10–100 Gb/s) và xác định chênh lệch công suất (Power penalty).
- **Bandwidth Sweep**: Khảo sát BER và EVM theo băng thông bộ thu ($B/R_s$), xác định 3 vùng hoạt động (giới hạn bởi ISI, vùng băng thông tối ưu, giới hạn bởi nhiễu tích lũy).
- **Dispersion Sweep**: Khảo sát BER theo hệ số tán sắc ($D$) và xác định giới hạn chịu tán sắc tối đa ($D_{limit}$) của OOK và 4-PAM.
- **Length Sweep & $L_{max}$**: Khảo sát BER theo chiều dài sợi quang ($L$) và so sánh cự ly truyền dẫn tối đa $L_{max}$ giữa OOK và 4-PAM theo tốc độ bit và công suất phát.

## Structure / Files

- `imdd_lib.py` — Thư viện mô phỏng cốt lõi, bao gồm khởi tạo các khối phát/kênh/thu, các hàm quét tham số 1D/2D, tính toán chỉ tiêu (BER, Q-factor, EVM) và trực quan hóa kết quả.
- `run_all_experiments.py` — Script chính tự động thi hành toàn bộ các khảo sát, xuất dữ liệu và lưu biểu đồ kết quả. (Cũ)
- `main_simulation.ipynb`/`main_simulation_new.ipynb` — Notebook minh họa chi tiết từng bước quy trình chạy mô phỏng, cấu hình tham số và trực quan hóa kết quả.
- `requirements.txt` — Danh sách các thư viện Python cần thiết.

## Cài đặt

```bash
pip install -r requirements.txt
pip install OptiCommPy      # thư viện core mô phỏng quang
```

## Chạy 

Toàn bộ các khảo sát và thử nghiệm mô phỏng được thực hiện trực tiếp thông qua hai file Jupyter Notebook: main_simulation.ipynb và main_simulation_new.ipynb

## Kết quả mô phỏng

Toàn bộ kết quả mô phỏng, hình ảnh biểu đồ đồ thị (PNG) và dữ liệu thống kê bảng biểu thu được sau khi thực thi notebook đều được lưu trong thư mục result/:

## Kết luận Trade-off chính

1. **Độ nhạy thu & Công suất**: OOK có độ nhạy thu tốt hơn 4-PAM (power penalty của 4-PAM khoảng 4 dB tại 10 Gb/s để đạt $\text{BER} = 10^{-3}$).
2. **Băng thông phần cứng**: 4-PAM tiết kiệm băng thông ký hiệu ($R_s = R_b/2$), giúp hoạt động tốt hơn OOK ở tốc độ bit cao khi phần cứng bị giới hạn băng thông.
3. **Khả năng chịu tán sắc**: Giới hạn chịu tán sắc $D_{limit}$ của 4-PAM cao hơn OOK khoảng 4 lần ở cùng tốc độ bit và chiều dài sợi.
4. **Cự ly truyền dẫn ($L_{max}$)**: OOK vượt trội ở tốc độ vừa và thấp ($10 - 20 \text{ Gb/s}$) trong tuyến bị giới hạn bởi suy hao. Ở tốc độ rất cao (vùng bị giới hạn bởi tán sắc), $L_{max}$ của cả hai định dạng đều suy giảm và hội tụ về mức thấp dưới $10 \text{ km}$.