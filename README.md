# DCVC-RT VCM: video-only machine-oriented coding

Dự án huấn luyện bộ mã hóa video dự đoán `DMC` của DCVC-RT cho tác vụ máy.
Không có `DMCI`, image pretraining, RGB/YUV pixel MSE, PSNR hay enhancement
layer dành cho người xem.

## Pipeline

```text
ảnh gốc x_t ──────> YOLOv5 teacher (frozen, no_grad) ──> {F4_t, F6_t, F9_t}
     │
     └─> DMC trainable ─> estimated BPP + ảnh giải nén x_hat_t
                                             │
                                             └─> YOLOv5 clone (frozen)
                                                       │
                                                       └─> {F4_hat_t, F6_hat_t, F9_hat_t}

D_l = MSE(F_l_t, F_l_hat_t)
D_machine = sum_l alpha_l * D_l
Loss_t = BPP_t + lambda(QP_base) * w_t * D_machine
```

Hai YOLOv5 extractor dùng cùng trọng số pretrained và luôn bị đóng băng. Nhánh
teacher chạy trong `no_grad`. Nhánh ảnh giải nén vẫn cho gradient đi qua input,
nhưng optimizer chỉ cập nhật DMC.

Mặc định Feature MSE lấy ba tầng backbone:

- Layer 4, stride 8: chi tiết không gian và vật thể nhỏ.
- Layer 6, stride 16: đặc trưng mức giữa.
- Layer 9, stride 32: ngữ nghĩa sâu.

## Protocol train theo bài DCVC-RT

Bài [Towards Practical Real-Time Neural Video Compression](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf)
công bố hai bước dữ liệu:

1. Train bằng các sequence 7 frame của Vimeo-90K.
2. Fine-tune bằng video Vimeo gốc đã xử lý thành sequence dài.

Dự án giữ cách train hai bước nhưng thay dữ liệu ở bước 2 bằng REDS
`train_sharp`/`val_sharp`, do original Vimeo không còn được phát hành thành một
gói ổn định. Vì vậy Stage 2 là protocol thay thế, không phải tái lập dataset
nguyên bản của paper.

Trong mỗi iteration, `QP_base` được lấy ngẫu nhiên trong `[0, 63]`. Lịch phân cấp
cho nhóm 8 ảnh là:

```text
vị trí ảnh:       0   1   2   3   4   5   6   7
QP offset:        0   8   0   4   0   4   0   4
distortion w_t: 0.5 1.2 0.5 0.9 0.5 1.2 0.5 0.9
```

Lambda được nội suy log từ 1 đến 768:

```text
lambda(q) = exp(log(1) + q/63 * (log(768) - log(1)))
```

Đây là bản chuyển đổi machine-oriented của protocol trên. Distortion RGB/YUV
của codec gốc được thay hoàn toàn bằng multi-level YOLO Feature MSE:

```text
Loss_t = BPP_t + lambda(QP_base) * w_t * D_machine_t
```

Frame 0 là reference seed từ bên ngoài, không nén và không tính loss. Stage
`vimeo7` dùng 7 vị trí đầu của lịch và mã hóa 6 P-frame. Stage `reds8` dùng đủ
8 vị trí và mã hóa 7 P-frame. Sau khi lấy trung bình loss các P-frame, chương
trình backpropagate một lần, clip gradient và cập nhật DMC bằng AdamW.

Theo ý tưởng progressive sequence training trong
[training recipe DCVC-UF](https://github.com/microsoft/DCVC/blob/main/training.md),
stage `vimeo7` không đưa ngay toàn bộ 7 frame vào từ epoch đầu:

```text
epoch 1–5:    2 frame = 1 seed + 1 P-frame
epoch 6–10:   3 frame = 1 seed + 2 P-frame
epoch 11–20:  5 frame = 1 seed + 4 P-frame
epoch 21+:    7 frame = 1 seed + 6 P-frame
```

Curriculum này giúp DMC thích nghi dần từ objective codec pretrained sang
Feature MSE trước khi học tích lũy sai số qua cả chuỗi. Mỗi lần validation luôn
dùng đủ 7 frame. Có thể đổi lịch bằng `--vimeo-curriculum-frames` và
`--vimeo-curriculum-start-epochs`. Mỗi lần validation, từng clip được kiểm tra ở
bốn `QP_base = {0, 21, 42, 63}`. Checkpoint `best.pt` được chọn theo trung bình
`BPP_estimated + lambda * Feature MSE` của cả bốn rate point, thay vì chỉ QP 32.
Bộ đếm early stopping và checkpoint `best.pt` được reset khi chuyển độ dài;
early stopping chỉ được phép kích hoạt sau khi đã đến pha 7 frame.

Optimizer là AdamW với `weight_decay=1e-4`. Weight decay chỉ áp dụng cho các
trọng số thông thường; các tham số quantization `q_*`, bias và tham số 1-D dùng
weight decay bằng 0 để không làm lệch rate-control.

Mã train DCVC-RT chính thức chưa được phát hành; paper không công bố số epoch,
batch size hay learning rate cụ thể. Các giá trị đó trong script là cấu hình có
thể điều chỉnh, không phải hyperparameter chính thức.

## Dataset stage 1: Vimeo-90K Septuplet

Tải Vimeo-90K Septuplet từ [trang dataset chính thức](http://toflow.csail.mit.edu/)
và giải nén theo cấu trúc:

```text
vimeo_septuplet/
├── sequences/
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── ...
│   │   │   └── im7.png
│   │   └── ...
│   └── ...
├── sep_trainlist.txt
└── sep_testlist.txt
```

Không cần nhãn detection để train vì teacher tự sinh target feature từ ảnh gốc.

## Dataset stage 2: REDS sharp sequences

Dự án dùng [REDS](https://seungjunnah.github.io/Datasets/reds.html) làm dataset
thay thế công khai cho original Vimeo. REDS có 240 sequence train và 30 sequence
validation; mỗi sequence chứa 100 frame RGB 720p liên tiếp. Chỉ cần tải
`train_sharp` và `val_sharp`; không dùng bản blur, low-resolution hoặc REDS4.

```text
REDS/
├── train_sharp/
│   ├── 000/
│   │   ├── 00000000.png
│   │   └── ...
│   └── 239/
└── val_sharp/
    ├── 000/
    │   ├── 00000000.png
    │   └── ...
    └── 029/
```

Loader tự tìm các thư mục sequence nên REDS không cần `train.txt`/`val.txt`.
`--samples-per-sequence 8` lấy tám temporal crop ngẫu nhiên từ mỗi sequence
trong một epoch. Đây là thay thế thực dụng cho dự án, không phải dataset
fine-tune nguyên bản được bài DCVC-RT công bố.

## Cài đặt

Cài PyTorch phù hợp với CUDA trước, sau đó:

```powershell
pip install -r requirements.txt
```

YOLOv5 có thể được tải online bằng Torch Hub hoặc dùng hoàn toàn offline:

```powershell
python train_vcm_final.py `
  ... `
  --yolov5-repo D:\models\yolov5-v7 `
  --yolov5-weights D:\models\yolov5s.pt
```

Xem hướng dẫn từng bước dành riêng cho Kaggle tại
[KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).

## Chạy train hai stage

Stage 1, train trên Vimeo-90K 7 frame:

```powershell
python train_vcm_final.py `
  --training-stage vimeo7 `
  --data-dir D:\data\vimeo_septuplet\sequences `
  --train-list D:\data\vimeo_septuplet\sep_trainlist.txt `
  --val-dir D:\data\vimeo_septuplet\sequences `
  --val-list D:\data\vimeo_septuplet\sep_testlist.txt `
  --video-init checkpoints\cvpr2025_video.pth.tar `
  --checkpoint-dir checkpoints\vcm_vimeo7
```

`--video-init` là tùy chọn. Dùng nó để machine-oriented fine-tune từ checkpoint
video DCVC-RT; bỏ tham số này nếu muốn train DMC từ khởi tạo ngẫu nhiên.

Stage 2, khởi tạo từ kết quả tốt nhất của stage 1 và fine-tune trên REDS:

```powershell
python train_vcm_final.py `
  --training-stage reds8 `
  --data-dir D:\data\REDS\train_sharp `
  --val-dir D:\data\REDS\val_sharp `
  --samples-per-sequence 8 `
  --video-init checkpoints\vcm_vimeo7\best.pt `
  --checkpoint-dir checkpoints\vcm_reds8
```

Script bắt buộc stage `reds8` phải có `--video-init` hoặc `--resume` để tránh
vô tình train chuỗi dài từ đầu.

Để tiếp tục đúng một stage sau khi bị dừng, dùng `--resume` thay vì
`--video-init`:

```powershell
python train_vcm_final.py `
  --training-stage reds8 `
  --data-dir D:\data\REDS\train_sharp `
  --resume checkpoints\vcm_reds8\latest.pt
```

Checkpoint tạo trước khi dự án chuyển sang AdamW không có optimizer state tương
thích. Với checkpoint cũ, dùng `--video-init`; chỉ dùng `--resume` cho checkpoint
schema mới do script hiện tại tạo ra. `latest.pt` được ghi đè nguyên tử mỗi
epoch và chứa cả DMC, AdamW, scheduler, best loss, early-stopping state và pha
curriculum hiện tại. `best.pt` chỉ được cập nhật khi validation tốt hơn.
Mặc định script lưu snapshot mỗi 10 epoch và chỉ giữ hai snapshot mới nhất.

Các mặc định quan trọng:

- `crop_size=256`
- `optimizer=AdamW`, `weight_decay=1e-4`
- Vimeo curriculum: `2 → 3 → 5 → 7` tại epoch `1, 6, 11, 21`
- `lambda_min=1`, `lambda_max=768`
- `validation_qps=[0, 21, 42, 63]`
- `validate_every=5`, `max_validation_batches=25`; mỗi clip được chạy ở cả 4 QP
- `save_every=10`, `keep_periodic_checkpoints=2`
- `feature_layer_indices=[4, 6, 9]`
- trọng số feature mặc định bằng nhau và được chuẩn hóa

GPU 4 GB có thể không đủ cho backpropagation xuyên nhiều P-frame với crop 256.
Khi OOM, giảm `--crop-size` xuống 128 hoặc 64 (vẫn phải chia hết cho 16), giữ
`--batch-size 1`, rồi mới cân nhắc gradient checkpointing/mixed precision.

## Evaluate

Evaluation dùng video đầy đủ ở độ phân giải gốc, bitstream entropy thật và
ground-truth bounding boxes:

```text
vcm_eval/
├── frames/
│   └── Kimono/
│       ├── 000000.png
│       └── ...
├── labels/
│   └── Kimono/
│       ├── 000000.txt
│       └── ...
└── manifest.json
```

Mỗi label dùng YOLO ground-truth format:

```text
class_id x_center y_center width height
```

Build entropy coder:

```powershell
cd src\cpp
pip install .
cd ..\..
```

Đo anchor với 4 rate points:

```powershell
python evaluate_vcm.py --mode codec `
  --data-dir D:\data\vcm_eval `
  --dataset-manifest manifest.json `
  --video-ckpt checkpoints\cvpr2025_video.pth.tar `
  --method-name dcvc_rt_anchor `
  --qps 0 21 42 63
```

Đo model machine-oriented trong cùng điều kiện:

```powershell
python evaluate_vcm.py --mode codec `
  --data-dir D:\data\vcm_eval `
  --dataset-manifest manifest.json `
  --video-ckpt checkpoints\vcm_reds8\best.pt `
  --method-name dcvc_rt_vcm `
  --qps 0 21 42 63
```

Tính BD-rate-mAP và vẽ RD curve:

```powershell
python evaluate_vcm.py --mode bdrate `
  --anchor-results output\evaluation\dcvc_rt_anchor_results.json `
  --candidate-results output\evaluation\dcvc_rt_vcm_results.json `
  --rate actual_bpp `
  --metric map5095
```

Train và evaluation là hai đường khác nhau:

```text
Train:       forward_train -> estimated BPP -> Feature MSE -> backward
Evaluation:  compress -> actual bitstream -> decompress -> ground-truth mAP
```

## Các file chính

```text
src/models/video_model.py       DMC video codec
src/models/vcm_loss.py          frozen YOLO + machine Feature MSE
src/utils/dataset.py            Vimeo-90K 7-frame + long-video loader
src/utils/vcm_eval_dataset.py   full-resolution frames + ground truth
src/utils/detection_map.py      mAP@0.5 và mAP@[0.5:0.95]
src/utils/vcm_bitstream.py      actual P-frame sequence container
train_vcm_final.py              two-stage variable-rate video training
evaluate_vcm.py                 actual BPP/kbps, mAP, BD-rate và RD curves
evaluate_hevc.py                HM HEVC encode/decode, actual rate và YOLO mAP
compare_codecs_bd_rate.py       so sánh HEVC/Learned Scalable/DCVC-RT VCM
```

Quy trình tạo HEVC anchor độc lập được mô tả tại
[HEVC_EVALUATION.md](HEVC_EVALUATION.md). Phần này chỉ dùng khi evaluation,
không thay đổi hoặc yêu cầu train lại model.
