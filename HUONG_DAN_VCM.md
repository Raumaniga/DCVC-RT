# Hướng dẫn Setup, Training & Evaluation (DCVC-RT VCM)

Tài liệu này hướng dẫn cách thiết lập môi trường, cách huấn luyện mô hình (Training), và phương pháp đánh giá (Evaluation) cho kiến trúc **DCVC-RT-Based Machine-Oriented Video Codec**.

---

## 1. Kiến trúc & Luồng hoạt động (Workflow)

Dự án này triển khai kiến trúc nén video tối ưu cho máy móc (Machine-Oriented Video Coding), dựa trên nền tảng DCVC-RT. 

### Sơ đồ luồng (Pipeline)

```text
Current Frame (x_t) → DCVC-RT Encoder → Q/AE → AD → DCVC-RT Decoder → Reconstruction Generation → Decoded Frame (f_t)
                                                                                                        ↓
                                                                                        Trainable Cloned CV Front End 
                                                                                                        ↓
                                                                                                      r̂_t^M
                                                                                                        ↓
x_t → Frozen Original CV Front End → r_t ───────────────────────────────────────────────→ D_task = MSE(r_t, r̂_t^M)
                                                                                                        ↓
                                                       L_M = (1/N) Σ [R(t) + λ_base * D_task(t)]
```

### Giải thích các thành phần:
- **DCVC-RT Core**: Bao gồm Encoder, Quantization/Entropy (Q/AE), Decoder. Nền tảng cốt lõi của nén video.
- **Task Back End & Original CV Front End (YOLOv5)**: Đóng vai trò tạo ra các feature gốc (`r_t`) đóng vai trò là "Ground Truth" (Nhãn chuẩn). Phần này **hoàn toàn đóng băng (Frozen)**, không tham gia vào quá trình cập nhật weights.
- **Trainable Cloned CV Front End**: Một bản sao của mạng trích xuất đặc trưng (Feature Extractor), nhưng được cấu hình để có thể học (**Trainable**). Nó trích xuất đặc trưng `r̂_t^M` từ frame đã giải nén (`f_t`).
- **Machine-Oriented Objective (L_M)**: Loss function được tối ưu hóa. Thay vì tối ưu mức độ sai lệch điểm ảnh (Pixel MSE - PSNR), mạng tối ưu hóa mức độ sai lệch đặc trưng (**Feature MSE**). Tổng loss là sự kết hợp giữa tỉ lệ nén (Rate `R(t)`) và độ méo mó đặc trưng (Distortion `D_task`).
- **Lưu ý**: Lớp Enhancement (Enhancement layer) đã được loại bỏ hoàn toàn theo sơ đồ.

---

## 2. Hướng dẫn cài đặt (Setup)

### Bước 1: Yêu cầu hệ thống
- Môi trường khuyến nghị: Google Colab (với T4 hoặc L4 GPU), hoặc máy tính có GPU NVIDIA.
- Python 3.10 trở lên.

### Bước 2: Tạo môi trường Conda (Nếu chạy trên Local)
```bash
conda create -n dcvc_vcm python=3.10
conda activate dcvc_vcm
```

### Bước 3: Cài đặt thư viện
Cài đặt PyTorch với CUDA hỗ trợ:
```bash
# Thay đổi phiên bản cu121/cu118 tuỳ vào CUDA version của bạn. Trên Colab thường đã có sẵn torch.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Cài đặt các thư viện khác trong requirements:
```bash
pip install -r requirements.txt
```

### Bước 4: Tải Pretrained Checkpoints (Baseline DCVC-RT)
Để quá trình training hội tụ nhanh, bạn cần sử dụng pretrained checkpoints của DCVC-RT (được train trên chuẩn RD truyền thống). Đặt chúng vào thư mục `checkpoints/`.
- `cvpr2025_image.pth.tar` (Dùng làm pretrain cho I-frame / DMCI)
- `cvpr2025_video.pth.tar` (Dùng làm pretrain cho P-frame / DMC)

---

## 3. Hướng dẫn Huấn luyện (Training)

Quá trình huấn luyện diễn ra qua 3 giai đoạn (Stages) bằng file `train_vcm_final.py`:

### Stage 1: Huấn luyện mạng nén ảnh (I-frame / DMCI)
Giai đoạn này đóng băng phần video, chỉ huấn luyện DMCI và lớp Trainable Cloned Front End bằng VCM Loss.
```bash
python train_vcm_final.py --stage 1 \
    --epochs 20 \
    --lambda_base 256 \
    --data_dir /path/to/vimeo_septuplet/sequences \
    --save_dir checkpoints/vcm
```
**Output mong đợi**: `checkpoints/vcm/vcm_dmci_best.pth`

### Stage 2: Huấn luyện mạng nén video (P-frame / DMC)
Giai đoạn này sử dụng model DMCI tốt nhất từ Stage 1 (và đóng băng nó). Chỉ cập nhật weights cho phần DMC và Trainable Front End.
```bash
python train_vcm_final.py --stage 2 \
    --epochs 15 \
    --lambda_base 256 \
    --dmci_ckpt checkpoints/vcm/vcm_dmci_best.pth \
    --data_dir /path/to/vimeo_septuplet/sequences \
    --save_dir checkpoints/vcm
```
**Output mong đợi**: `checkpoints/vcm/vcm_dmc_best.pth`

### Stage 3: Fine-tune toàn bộ (Joint Training)
Huấn luyện đồng thời cả DMCI và DMC để tối ưu toàn cục.
```bash
python train_vcm_final.py --stage 3 \
    --epochs 5 \
    --lambda_base 256 \
    --dmci_ckpt checkpoints/vcm/vcm_dmci_best.pth \
    --dmc_ckpt checkpoints/vcm/vcm_dmc_best.pth \
    --data_dir /path/to/vimeo_septuplet/sequences \
    --save_dir checkpoints/vcm
```
**Output mong đợi**: `checkpoints/vcm/vcm_dmci_joint_best.pth` và `checkpoints/vcm/vcm_dmc_joint_best.pth`

*(Lưu ý: Có thể dùng vòng lặp để train cho nhiều mức độ chất lượng (λ) khác nhau như 64, 128, 256, 512 nhằm tạo ra đường cong Rate-Distortion (RD curve).)*

---

## 4. Đánh giá (Evaluation)

Script `evaluate_vcm.py` xử lý toàn bộ quá trình kiểm tra mô hình. Nó bao gồm 3 chế độ (mode):

### Phân tích Loss Curves (Mode: `training`)
Đọc file log (CSV) trong quá trình train và vẽ biểu đồ biến thiên của Loss, BPP, và Feature MSE để đánh giá độ hội tụ và rủi ro overfitting.
```bash
python evaluate_vcm.py --mode training --log_dir checkpoints/vcm/logs/
```

### Đánh giá Performance Đa mức chất lượng (Mode: `vcm`)
Kiểm tra BPP, Pixel PSNR, và đặc biệt là **Feature MSE** tại các mức Quantization Parameter (QP) khác nhau.
```bash
python evaluate_vcm.py --mode vcm \
    --dmci_ckpt checkpoints/vcm/vcm_dmci_joint_best.pth \
    --dmc_ckpt checkpoints/vcm/vcm_dmc_joint_best.pth \
    --data_dir /path/to/vimeo_septuplet/sequences \
    --qp_list 0 16 32 48 \
    --output_dir output/evaluation
```
Kết quả được xuất ra file JSON (ví dụ: `vcm_results.json`) và biểu đồ (BPP vs Feature MSE).

### So sánh BD-Rate (Mode: `bdrate`)
Tính toán **Bjøntegaard Delta Rate (BD-Rate)**. Metric này so sánh hai đường RD Curves (ví dụ Baseline vs VCM Model mới).
- **BD-Rate < 0**: Có nghĩa là model mới tiết kiệm được băng thông (bitrate) nhưng vẫn giữ nguyên chất lượng (Feature MSE tương đương).
- **BD-Rate > 0**: Model mới bị tiêu tốn nhiều bitrate hơn.

Tính toán tự động từ 2 file JSON (Kết quả của model gốc vs model mới train):
```bash
python evaluate_vcm.py --mode bdrate \
    --anchor_results output/evaluation/anchor_results.json \
    --test_results output/evaluation/vcm_results.json
```
*(Lưu ý: Thuật toán tính BD-Rate được đặt trong `src/utils/bd_rate.py` sử dụng phương pháp nội suy PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) chuẩn mực của ITU-T VCEG).*
