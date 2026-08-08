# Huấn luyện DCVC-RT VCM từ đầu đến cuối trên Kaggle

Hướng dẫn này dành cho pipeline video-only, machine-oriented:

```text
estimated BPP + multi-level YOLOv5 Feature MSE
```

Không train DMCI, không dùng RGB/YUV pixel MSE và không cần build entropy coder
C++ trong quá trình train.

## 1. Tạo các Kaggle Dataset đầu vào

Gắn ít nhất hai dataset vào notebook.

### Vimeo-90K Septuplet

```text
/kaggle/input/vimeo90k/
└── vimeo_septuplet/
    ├── sequences/
    │   └── 00001/
    │       └── 0001/
    │           ├── im1.png
    │           ├── ...
    │           └── im7.png
    ├── sep_trainlist.txt
    └── sep_testlist.txt
```

### Checkpoint video DCVC-RT

```text
/kaggle/input/dcvc-rt-weights/
└── cvpr2025_video.pth.tar
```

### Tùy chọn: YOLOv5 dùng offline

Nếu notebook không được bật Internet, tạo thêm dataset:

```text
/kaggle/input/yolov5-v7-offline/
├── yolov5/                 # checkout tag v7.0
│   ├── hubconf.py
│   ├── models/
│   └── ...
└── yolov5s.pt
```

Stage 2 dùng REDS sharp sequences:

```text
/kaggle/input/reds-dataset/
└── REDS/
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

Tải bản REDS đầy đủ gồm `train_sharp` và `val_sharp`, không dùng REDS4.
Mỗi sequence REDS có 100 frame nên đáp ứng clip 8 frame của Stage 2.
Nguồn tải và hướng dẫn cấu trúc:
[REDS chính thức](https://seungjunnah.github.io/Datasets/reds.html) và
[BasicSR Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#reds).
`reds-dataset` chỉ là slug minh họa; thay nó bằng slug Kaggle Dataset của bạn.

## 2. Tạo notebook và bật GPU

Trong Notebook Settings:

1. Chọn GPU accelerator.
2. Bật Internet nếu muốn Torch Hub tự tải YOLOv5 v7 và `yolov5s.pt`.
3. Gắn các Kaggle Dataset ở bước 1.

Kiểm tra GPU:

```python
import torch

assert torch.cuda.is_available(), "Notebook chưa bật GPU"
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM MiB:",
    torch.cuda.get_device_properties(0).total_memory / 2**20,
)
```

## 3. Clone source mới nhất

```bash
%cd /kaggle/working
!git clone https://github.com/uetot1/DCVC-RT.git
%cd /kaggle/working/DCVC-RT
```

Kiểm tra commit:

```bash
!git log -1 --oneline
```

## 4. Cài dependency

Kaggle đã có PyTorch/CUDA. Không nên cài lại PyTorch trừ khi phiên bản hiện tại
bị lỗi. Chỉ cài các dependency còn thiếu:

```bash
!pip install -q \
  tqdm scipy matplotlib pybind11 \
  opencv-python-headless ultralytics
```

Kiểm tra import:

```python
import cv2
import numpy
import scipy
import tqdm
import torch
import torchvision

print("dependency_check=PASS")
```

## 5. Kiểm tra đường dẫn dữ liệu

```python
from pathlib import Path

VIMEO_ROOT = Path("/kaggle/input/vimeo90k/vimeo_septuplet")
VIDEO_CKPT = Path(
    "/kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar"
)

assert (VIMEO_ROOT / "sequences").is_dir()
assert (VIMEO_ROOT / "sep_trainlist.txt").is_file()
assert (VIMEO_ROOT / "sep_testlist.txt").is_file()
assert VIDEO_CKPT.is_file()

first_id = (
    (VIMEO_ROOT / "sep_trainlist.txt")
    .read_text()
    .splitlines()[0]
    .strip()
)
first_sequence = VIMEO_ROOT / "sequences" / first_id
print("First sequence:", first_sequence)
print("Frames:", sorted(path.name for path in first_sequence.glob("*.png")))
```

Kết quả phải có đúng `im1.png` đến `im7.png`.

## 6. Chuẩn bị YOLOv5

### Cách A — notebook có Internet

Chạy một lần để cache repository và weights:

```python
import torch
from src.models.yolov5_extractor import load_yolov5

# Do not call torch.hub.load directly here. PyTorch >= 2.6 defaults
# torch.load(weights_only=True), while the trusted YOLOv5 v7 checkpoint
# contains the legacy models.yolo.Model object. The project loader applies
# weights_only=False only while loading this pinned YOLOv5 checkpoint.
model = load_yolov5("yolov5s")
del model
torch.cuda.empty_cache()
print("YOLOv5 online cache=PASS")
```

Khi train không cần truyền `--yolov5-repo` hoặc `--yolov5-weights`.

### Cách B — notebook offline

Thêm vào mọi lệnh train:

```bash
--yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
--yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt
```

Script chỉ load YOLO một lần rồi sao chép backbone. Teacher đóng băng; clone giữ
BatchNorm ở eval nhưng trọng số được optimizer học chung với DMC. `latest.pt`,
`best.pt` và `epoch_N.pt` đều lưu cloned front-end để evaluation dùng đúng mạng
đã train.

## 7. Chạy smoke test trước

Không bắt đầu train dài trước khi lệnh này hoàn thành:

```bash
!python train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --checkpoint-dir /kaggle/working/checkpoints/smoke_vimeo7 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 1 \
  --max-batches 5 \
  --max-validation-batches 2 \
  --validate-every 1 \
  --save-every 0
```

Smoke test đạt khi:

- Có 5 batch với loss/BPP hữu hạn.
- `skipped_batches=0`.
- Không CUDA OOM.
- Có `latest.pt`, `best.pt` và CSV log.

Kiểm tra:

```bash
!find /kaggle/working/checkpoints/smoke_vimeo7 -maxdepth 2 -type f
!nvidia-smi
```

## 8. Stage 1 — Vimeo-90K machine adaptation

### Chạy toàn bộ dataset mỗi epoch

Đây là cấu hình gần protocol dataset đầy đủ nhất, nhưng có thể quá dài cho một
Kaggle session:

```bash
!python train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_vimeo7 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --validate-every 5 \
  --validation-qps 0 21 42 63 \
  --max-validation-batches 25 \
  --save-every 10 \
  --keep-periodic-checkpoints 2
```

### Cấu hình thực tế hơn cho quota Kaggle

Giới hạn số update trong một epoch để mỗi epoch chắc chắn kết thúc và tạo
`latest.pt`. DataLoader shuffle lại toàn bộ Vimeo-90K ở epoch tiếp theo:

```bash
!python train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_vimeo7 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --max-batches 1000 \
  --validate-every 5 \
  --validation-qps 0 21 42 63 \
  --max-validation-batches 25 \
  --save-every 10 \
  --keep-periodic-checkpoints 2
```

`--max-batches 1000` là cấu hình vận hành cho Kaggle, không phải con số được
paper công bố. Sau smoke test, đo thời gian 100 batch rồi điều chỉnh để một epoch
hoàn thành an toàn trong session.

Curriculum mặc định:

```text
epoch 1–5:    2 frame
epoch 6–10:   3 frame
epoch 11–20:  5 frame
epoch 21+:    7 frame
```

## 9. Resume Stage 1

Nếu kernel vẫn giữ `/kaggle/working`:

```bash
!python train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --resume /kaggle/working/checkpoints/vcm_vimeo7/latest.pt \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_vimeo7 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --max-batches 1000
```

Nếu bắt đầu session mới:

1. Save Version hoặc tạo Kaggle Dataset từ output session trước.
2. Gắn output đó làm input cho notebook mới.
3. Truyền đường dẫn input `latest.pt` vào `--resume`.
4. Vẫn ghi checkpoint mới vào `/kaggle/working`.

Ví dụ:

```bash
--resume /kaggle/input/vcm-vimeo7-previous/vcm_vimeo7/latest.pt
```

Không truyền đồng thời `--resume` và `--video-init`.

## 10. Chọn checkpoint Stage 1

Sau khi hoàn thành:

```text
/kaggle/working/checkpoints/vcm_vimeo7/
├── latest.pt
├── best.pt
├── epoch_*.pt             # tối đa hai snapshot mặc định
└── logs/
```

- `latest.pt`: tiếp tục đúng optimizer, scheduler, curriculum và early stopping.
- `best.pt`: dùng để khởi tạo Stage 2 hoặc evaluation.
- `epoch_*.pt`: snapshot dự phòng.

## 11. Stage 2 — fine-tune REDS sharp sequences

Stage 2 luôn dùng 8 frame: 1 external seed + 7 P-frame.

Kiểm tra REDS trước:

```python
from pathlib import Path

REDS_ROOT = Path("/kaggle/input/reds-dataset/REDS")
train_sequences = sorted(
    path for path in (REDS_ROOT / "train_sharp").iterdir()
    if path.is_dir()
)
val_sequences = sorted(
    path for path in (REDS_ROOT / "val_sharp").iterdir()
    if path.is_dir()
)

assert len(train_sequences) == 240
assert len(val_sequences) == 30
assert len(list(train_sequences[0].glob("*.png"))) == 100
print("REDS check=PASS")
```

```bash
!python train_vcm_final.py \
  --training-stage reds8 \
  --data-dir /kaggle/input/reds-dataset/REDS/train_sharp \
  --val-dir /kaggle/input/reds-dataset/REDS/val_sharp \
  --samples-per-sequence 8 \
  --video-init /kaggle/working/checkpoints/vcm_vimeo7/best.pt \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_reds8 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --max-batches 1000 \
  --validate-every 5 \
  --validation-qps 0 21 42 63 \
  --max-validation-batches 25 \
  --save-every 10 \
  --keep-periodic-checkpoints 2
```

Stage 2 resume:

```bash
!python train_vcm_final.py \
  --training-stage reds8 \
  --data-dir /kaggle/input/reds-dataset/REDS/train_sharp \
  --val-dir /kaggle/input/reds-dataset/REDS/val_sharp \
  --samples-per-sequence 8 \
  --resume /kaggle/working/checkpoints/vcm_reds8/latest.pt \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_reds8 \
  --crop-size 128 \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 100 \
  --max-batches 1000
```

Không dùng `--resume` từ Stage 1 cho Stage 2. Chuyển stage phải dùng
`--video-init .../vcm_vimeo7/best.pt`. Tên stage cũ `long8` vẫn được chấp nhận
như alias để resume checkpoint cũ, nhưng run mới nên dùng `reds8`.

Validation mặc định dùng bốn `QP_base = {0, 21, 42, 63}`. Mỗi validation clip
được chạy bốn lần, nên `--max-validation-batches 25` tương ứng 100 lượt chạy
codec. `best.pt` tối thiểu hóa mean validation loss của bốn rate point. Đây là
estimated BPP + Feature MSE trên tập validation, không phải actual-bitstream mAP;
mAP và BD-rate vẫn được đo riêng bằng `evaluate_vcm.py` sau khi train.

Mặc định script dùng thiết kế lai Learned Scalable + TransTIC-inspired:

```bash
--train-cloned-frontend \
--feature-layer-indices 4 6 9 \
--feature-layer-weights 1 1 1
```

Ba trọng số được chuẩn hóa thành `1/3`. Đây không phải bộ trọng số đã được paper
chứng minh tối ưu cho YOLOv5; hãy giữ các periodic checkpoint và chọn bằng
BD-rate–mAP trên validation set có nhãn. Không chọn bằng Class D test set.

Checkpoint cũ không chứa clone không thể `--resume` vào optimizer mới. Có thể
dùng checkpoint đó qua `--video-init`: DMC được kế thừa, còn clone khởi tạo từ
YOLO pretrained rồi bắt đầu joint training.

## 12. Xử lý CUDA OOM

Thực hiện theo thứ tự:

1. Giữ `--batch-size 1`.
2. Giảm `--crop-size 128` xuống `96`.
3. Nếu vẫn OOM, giảm xuống `64`.
4. Đảm bảo crop size luôn chia hết cho 16.
5. Không tăng `num_workers` để sửa OOM GPU; worker chỉ ảnh hưởng CPU/RAM.

Không tự bật AMP cho rate path vì entropy likelihood và CDF có thể mất ổn định
ở FP16.

## 13. Theo dõi train

```bash
!nvidia-smi
!ls -lh /kaggle/working/checkpoints/vcm_vimeo7
!tail -n 5 /kaggle/working/checkpoints/vcm_vimeo7/logs/*.csv
```

Các dấu hiệu cần dừng:

- Loss hoặc BPP liên tục tăng vô hạn.
- `skipped_batches` tăng thường xuyên.
- Validation Feature MSE không cải thiện qua nhiều lần kiểm tra.
- GPU OOM lặp lại.

Một batch NaN/Inf được bỏ qua và không cập nhật AdamW. Nếu tất cả batch trong
epoch đều bị bỏ qua, script dừng với lỗi thay vì tạo checkpoint hỏng.

## 14. Lưu kết quả trước khi kết thúc session

Chỉ cần giữ:

```text
latest.pt
best.pt
hai epoch snapshot mới nhất
logs/
```

Các file được ghi trong `/kaggle/working`, vì vậy hãy Save Version, tải xuống
hoặc tạo Kaggle Dataset output trước khi xóa session.

## 15. Sau khi train

Dùng `best.pt` của Stage 2 cho evaluation bốn rate point:

```text
actual BPP → mAP@0.5 / mAP@[0.5:0.95] → RD curve → BD-rate
```

Evaluation bằng actual bitstream mới cần build `src/cpp`; training không cần.
`evaluate_vcm.py` tự đọc cloned front-end từ checkpoint schema 8 và ghép nó với
YOLO task back-end đóng băng. Nếu log báo fallback sang pretrained front-end thì
đó là checkpoint legacy, chưa phải joint-trained Learned-Scalable-style model.
