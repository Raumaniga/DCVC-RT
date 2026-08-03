# DCVC-RT VCM: video-only machine-oriented coding

Dự án chỉ huấn luyện bộ mã hóa video dự đoán `DMC` của DCVC-RT cho tác vụ
machine-oriented. Không có `DMCI`, image codec, giai đoạn image pretrain hay
checkpoint ảnh.

## Pipeline

```text
ảnh gốc x_t ───────> YOLOv5 teacher (frozen, no_grad) ──> {F4_t, F6_t, F9_t}
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

Hai YOLOv5 extractor có cùng trọng số pretrained và luôn ở chế độ evaluation.
Nhánh teacher chạy trong `no_grad`. Nhánh ảnh giải nén vẫn giữ đường gradient
theo input, vì vậy gradient đi xuyên qua extractor nhưng chỉ cập nhật DMC.

Feature distortion mặc định lấy ba mức backbone YOLOv5:

- Layer 4, stride 8: chi tiết không gian và vật thể nhỏ.
- Layer 6, stride 16: cấu trúc/ngữ nghĩa mức giữa.
- Layer 9, stride 32: ngữ nghĩa sâu và vật thể lớn.

MSE được tính độc lập trên mỗi tensor nên không cần resize hoặc nối các feature
map khác kích thước. Các trọng số `alpha_l` được chuẩn hóa để có tổng bằng 1;
mặc định ba tầng có trọng số bằng nhau. CSV log ghi cả loss tổng hợp và
`feature_mse_l4`, `feature_mse_l6`, `feature_mse_l9`.

`DMC` là P-frame codec. Ảnh số 0 trong mỗi clip là reference seed do bên ngoài
cung cấp; nó không được nén, không tính rate và không tính loss. Bảy ảnh còn lại
được mã hóa nối tiếp và trạng thái DPB được truyền từ ảnh trước sang ảnh sau.

## Cơ chế train DCVC-RT

Mỗi iteration dùng đúng một nhóm 8 ảnh:

```text
vị trí ảnh:       0  1  2  3  4  5  6  7
QP offset:        0  8  0  4  0  4  0  4
distortion w_t: 0.5 1.2 0.5 0.9 0.5 1.2 0.5 0.9
```

Quy trình của một iteration:

1. Lấy ngẫu nhiên `QP_base` nguyên trong `[0, 63]`.
2. Ảnh tại vị trí `t` dùng `QP_t = QP_base + QP_offset[t]`. Vì có offset `+8`,
   DMC sử dụng đầy đủ index QP đến 71 như kiến trúc gốc.
3. Tính lambda bằng nội suy log:

   ```text
   lambda(q) = exp(log(1) + q/63 * (log(768) - log(1)))
   ```

4. Chỉ nhân `w_t` vào distortion; rate không bị nhân trọng số:

   ```text
   Loss_t = BPP_t + lambda(QP_base) * w_t * D_machine_t
   ```

5. Lấy trung bình loss của bảy P-frame, backpropagation một lần, clip gradient
   rồi cập nhật DMC bằng Adam.

Không có RGB MSE, YUV MSE, pixel MSE, PSNR hoặc perceptual loss trong objective.
Ảnh chỉ được đưa về tensor ba kênh theo giao diện đầu vào của codec và YOLO;
không có gradient nào đến từ chất lượng hiển thị cho người xem.

## Dữ liệu

Lịch train trên cần 8 ảnh liên tiếp. Vimeo-90K septuplet chuẩn chỉ có 7 ảnh nên
không thể tái hiện chính xác lịch này. Hãy dùng các video Vimeo gốc đã xử lý
thành thư mục frame dài, giống giai đoạn fine-tune của DCVC-FM/DCVC-RT:

```text
long_vimeo/
├── clip_0001/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── clip_0002/
│   └── ...
└── train.txt                 # tùy chọn
```

Mỗi dòng trong list là đường dẫn sequence tương đối với `--data-dir`. Nếu không
truyền list, loader tự tìm các thư mục chứa ảnh. Khi train, loader chọn ngẫu
nhiên một đoạn 8 ảnh, crop đồng nhất và có horizontal flip. Khi validation,
loader lấy 8 ảnh đầu và center crop để kết quả ổn định.

## Cài đặt

Cài bản PyTorch phù hợp với CUDA trước, sau đó:

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python train_vcm_final.py `
  --data-dir D:\data\long_vimeo `
  --train-list train.txt `
  --gop-size 8 `
  --video-init checkpoints\cvpr2025_video.pth.tar `
  --checkpoint-dir checkpoints\vcm_video
```

Các giá trị mặc định quan trọng:

- `lambda_min=1`, `lambda_max=768`
- `validation_qp=32`
- `feature_layer_indices=[4, 6, 9]`
- `feature_layer_weights` bằng nhau và được chuẩn hóa thành `[1/3, 1/3, 1/3]`

`--video-init` chỉ nhận checkpoint DMC/video. Không có checkpoint ảnh.

Có thể đổi tầng và trọng số, ví dụ ưu tiên ngữ nghĩa sâu hơn:

```powershell
python train_vcm_final.py `
  --data-dir D:\data\long_vimeo `
  --feature-layer-indices 4 6 9 `
  --feature-layer-weights 0.2 0.3 0.5
```

## Evaluate

Evaluation dùng video đầy đủ ở độ phân giải gốc, bitstream entropy thật và
ground-truth bounding boxes. Tạo cấu trúc:

```text
vcm_eval/
├── frames/
│   └── Kimono/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
├── labels/
│   └── Kimono/
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
└── manifest.json
```

Mỗi label dùng YOLO ground-truth format chuẩn hóa:

```text
class_id x_center y_center width height
```

Phải có file label cho mọi frame; dùng file rỗng nếu frame không có object.
`class_id` trong label phải dùng đúng class-index mapping của `--task-model`;
không được so sánh trực tiếp hai hệ class khác nhau.
`manifest.json`:

```json
{
  "sequences": [
    {
      "name": "Kimono",
      "frames_dir": "frames/Kimono",
      "labels_dir": "labels/Kimono",
      "fps": 24
    }
  ]
}
```

Có thể sao chép [dataset_manifest_example.json](dataset_manifest_example.json)
làm mẫu và thay đường dẫn sequence thực tế.

Build entropy coder trước khi đo bitstream thật:

```powershell
cd src\cpp
pip install .
cd ..\..
```

Đo anchor DCVC-RT gốc với 4 rate points:

```powershell
python evaluate_vcm.py --mode codec `
  --data-dir D:\data\vcm_eval `
  --dataset-manifest manifest.json `
  --video-ckpt checkpoints\cvpr2025_video.pth.tar `
  --method-name dcvc_rt_anchor `
  --qps 0 21 42 63
```

Bốn QP có thể điều chỉnh sau một lần chạy thử, nhưng phải giữ đúng bốn điểm,
có bitrate tăng và mAP tăng nghiêm ngặt, đồng thời curve anchor và candidate
phải có vùng mAP giao nhau để BD-rate hợp lệ.

Đo model machine-oriented với cùng điều kiện:

```powershell
python evaluate_vcm.py --mode codec `
  --data-dir D:\data\vcm_eval `
  --dataset-manifest manifest.json `
  --video-ckpt checkpoints\vcm_video\best.pt `
  --method-name dcvc_rt_vcm `
  --qps 0 21 42 63
```

Tính BD-rate-mAP và tạo RD curves:

```powershell
python evaluate_vcm.py --mode bdrate `
  --anchor-results output\evaluation\dcvc_rt_anchor_results.json `
  --candidate-results output\evaluation\dcvc_rt_vcm_results.json `
  --rate actual_bpp `
  --metric map5095
```

Kết quả gồm:

- Actual BPP từ kích thước file `.bin`, có tính container headers.
- Actual kbps từ số bit, FPS và số P-frame được mã hóa.
- Ground-truth `mAP@0.5` và `mAP@[0.5:0.95]`.
- `BD-rate-mAP`; giá trị âm là tiết kiệm bitrate tại cùng mAP.
- `rd_curve_actual_bpp_map50.png`.
- `rd_curve_actual_bpp_map5095.png`.
- `rd_points.csv`.

Train và evaluation là hai đường khác nhau:

```text
Train:       forward_train -> estimated BPP -> Feature MSE -> backward
Evaluation:  compress -> actual bitstream -> decompress -> ground-truth mAP
```

Bitstream protocol hiện là DMC-only: frame 0 là external seed và bị loại khỏi
cả rate lẫn mAP. Anchor và candidate bắt buộc dùng cùng protocol này.

## Các file chính

```text
src/models/video_model.py       DMC video codec
src/models/vcm_loss.py          frozen YOLO + machine Feature MSE
src/utils/dataset.py            contiguous long-video clip loader
src/utils/vcm_eval_dataset.py   full-resolution frames + ground truth
src/utils/detection_map.py      mAP@0.5 và mAP@[0.5:0.95]
src/utils/vcm_bitstream.py      actual P-frame sequence container
train_vcm_final.py              GOP-8 variable-rate training
evaluate_vcm.py                 actual BPP/kbps, mAP, BD-rate và RD curves
```
