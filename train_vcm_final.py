"""
DCVC-RT Machine-Oriented Video Codec — Final Training Script
=============================================================
Script huấn luyện cuối cùng cho DCVC-RT Based Machine-Oriented Video Codec.

Kết hợp:
  - Multi-stage video training (DMCI → DMC → Joint) từ train_video.py
  - Machine-Oriented VCM Loss (feature MSE thay vì pixel MSE) từ train_base_layer.py
  - Đúng theo sơ đồ kiến trúc "Proposed DCVC-RT-Based Machine-Oriented Video Codec"

Kiến trúc training (theo sơ đồ):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  x_t → DCVC-RT Encoder → Q/AE → AD → Decoder → Recon Gen → f_t       │
  │                                                      ↓                  │
  │                                        Trainable Cloned CV Front End    │
  │                                              → r̂_t^M                   │
  │                                                      ↓                  │
  │  x_t → Frozen Original CV Front End → r_t → D_task = MSE(r_t, r̂_t^M)  │
  │                                                      ↓                  │
  │         L_M = (1/N) Σ_t [R(t) + λ_base × D_task(t)]                   │
  └──────────────────────────────────────────────────────────────────────────┘

  Gradient chạy qua: DCVC-RT core + Trainable Cloned CV Front End
  Frozen (không gradient): Original CV Front End + Task Back End

3 giai đoạn huấn luyện:
  Stage 1: Train DMCI (I-frame)      — chỉ frame đầu tiên, VCM Loss
  Stage 2: Train DMC  (P-frame)      — đóng băng DMCI, VCM Loss
  Stage 3: Joint fine-tune DMCI+DMC  — cả hai cùng lúc, VCM Loss

Sử dụng:
  python train_vcm_final.py --stage 1 --epochs 20 --lambda_base 256 --data_dir /path/to/vimeo
  python train_vcm_final.py --stage 2 --epochs 15 --dmci_ckpt checkpoints/vcm/dmci_best.pth
  python train_vcm_final.py --stage 3 --epochs 5  --dmci_ckpt ... --dmc_ckpt ...
"""

import os
import csv
import json
import argparse
import random
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.models.vcm_loss import VCMLoss


# ============================================================================
# Training Logger — Ghi nhận metrics cho đánh giá
# ============================================================================

class TrainingLogger:
    """
    Logger ghi nhận training metrics vào CSV cho việc phân tích sau này.

    Ghi nhận:
      - Per-epoch: avg loss, avg bpp, avg feature_mse, avg pixel_psnr, lr
      - Per-batch: (tùy chọn) loss, bpp, feature_mse cho từng batch
    """

    def __init__(self, log_dir, stage, lambda_base):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"stage{stage}_lambda{lambda_base}_{timestamp}"

        # Epoch-level log
        self.epoch_log_path = os.path.join(log_dir, f"{prefix}_epoch.csv")
        self.epoch_file = open(self.epoch_log_path, 'w', newline='')
        self.epoch_writer = csv.writer(self.epoch_file)
        self.epoch_writer.writerow([
            'epoch', 'lr',
            'train_total_loss', 'train_rate_bpp', 'train_feature_mse',
            'train_distortion_weighted', 'train_pixel_psnr',
            'val_total_loss', 'val_rate_bpp', 'val_feature_mse',
            'val_distortion_weighted', 'val_pixel_psnr',
        ])

        # Batch-level log
        self.batch_log_path = os.path.join(log_dir, f"{prefix}_batch.csv")
        self.batch_file = open(self.batch_log_path, 'w', newline='')
        self.batch_writer = csv.writer(self.batch_file)
        self.batch_writer.writerow([
            'epoch', 'batch', 'total_loss', 'rate_bpp', 'feature_mse',
            'distortion_weighted', 'pixel_psnr',
        ])

        # Training config log
        self.config_path = os.path.join(log_dir, f"{prefix}_config.json")

    def log_config(self, args):
        """Lưu cấu hình training."""
        config = vars(args) if hasattr(args, '__dict__') else dict(args)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log_batch(self, epoch, batch_idx, metrics):
        """Ghi log cho mỗi batch."""
        self.batch_writer.writerow([
            epoch, batch_idx,
            f"{metrics['total_loss']:.6f}",
            f"{metrics['rate_bpp']:.6f}",
            f"{metrics['feature_mse']:.8f}",
            f"{metrics['distortion_weighted']:.6f}",
            f"{metrics['pixel_psnr']:.2f}",
        ])
        self.batch_file.flush()

    def log_epoch(self, epoch, lr, train_metrics, val_metrics=None):
        """Ghi log cho mỗi epoch."""
        row = [
            epoch,
            f"{lr:.8f}",
            f"{train_metrics['total_loss']:.6f}",
            f"{train_metrics['rate_bpp']:.6f}",
            f"{train_metrics['feature_mse']:.8f}",
            f"{train_metrics['distortion_weighted']:.6f}",
            f"{train_metrics['pixel_psnr']:.2f}",
        ]
        if val_metrics:
            row.extend([
                f"{val_metrics['total_loss']:.6f}",
                f"{val_metrics['rate_bpp']:.6f}",
                f"{val_metrics['feature_mse']:.8f}",
                f"{val_metrics['distortion_weighted']:.6f}",
                f"{val_metrics['pixel_psnr']:.2f}",
            ])
        else:
            row.extend(['', '', '', '', ''])
        self.epoch_writer.writerow(row)
        self.epoch_file.flush()

    def close(self):
        self.epoch_file.close()
        self.batch_file.close()
        self._plot_training_curves()
        
    def _plot_training_curves(self):
        """Tự động vẽ biểu đồ Loss Curves khi kết thúc training (nếu có matplotlib)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = pd.read_csv(self.epoch_log_path)
            if len(df) == 0:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Training Curves: {os.path.basename(self.epoch_log_path)}', fontsize=14, fontweight='bold')
            
            # Loss
            axes[0, 0].plot(df['epoch'], df['train_total_loss'], 'b-o', label='Train')
            if 'val_total_loss' in df.columns and df['val_total_loss'].notna().any():
                axes[0, 0].plot(df['epoch'], df['val_total_loss'], 'r-s', label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # BPP
            axes[0, 1].plot(df['epoch'], df['train_rate_bpp'], 'b-o', label='Train')
            if 'val_rate_bpp' in df.columns and df['val_rate_bpp'].notna().any():
                axes[0, 1].plot(df['epoch'], df['val_rate_bpp'], 'r-s', label='Val')
            axes[0, 1].set_title('Rate (BPP)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Feature MSE
            axes[1, 0].plot(df['epoch'], df['train_feature_mse'], 'b-o', label='Train')
            if 'val_feature_mse' in df.columns and df['val_feature_mse'].notna().any():
                axes[1, 0].plot(df['epoch'], df['val_feature_mse'], 'r-s', label='Val')
            axes[1, 0].set_title('Feature MSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # PSNR
            axes[1, 1].plot(df['epoch'], df['train_pixel_psnr'], 'b-o', label='Train')
            if 'val_pixel_psnr' in df.columns and df['val_pixel_psnr'].notna().any():
                axes[1, 1].plot(df['epoch'], df['val_pixel_psnr'], 'r-s', label='Val')
            axes[1, 1].set_title('Pixel PSNR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plot_path = self.epoch_log_path.replace('.csv', '.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  ✓ Đã tự động vẽ biểu đồ training tại: {plot_path}")
        except Exception as e:
            print(f"  ℹ️ Không thể tự động vẽ biểu đồ (Cần cài matplotlib & pandas): {e}")


# ============================================================================
# Validation — Chạy trên tập validation để kiểm tra overfitting
# ============================================================================

@torch.no_grad()
def validate_stage1(model, criterion, dataloader, device, qp_list=None):
    """Validation cho Stage 1 (DMCI I-frame).
    
    QP được bốc ngẫu nhiên từ [0, 63] mỗi batch, giống hệt phân phối
    trong training loop, để đảm bảo train_loss và val_loss có thể so sánh
    trực tiếp trên cùng một phân phối QP.
    """
    model.eval()

    accum = {
        'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
        'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
    }
    count = 0

    for frames in dataloader:
        frames = frames.to(device)
        x = frames[:, 0, :, :, :]
        qp = random.randint(0, 63)  # Cùng phân phối với training

        x_hat, rate_bpp = model.forward_train(x, qp)
        details = criterion(x, x_hat, rate_bpp, return_details=True)

        for k in accum:
            accum[k] += details[k].item()
        count += 1

    for k in accum:
        accum[k] /= max(count, 1)

    model.train()
    return accum


@torch.no_grad()
def validate_stage2(dmci, dmc, criterion, dataloader, device, qp_list=None):
    """Validation cho Stage 2 (DMC P-frame, DMCI frozen).
    
    QP được bốc ngẫu nhiên từ [0, 63] mỗi batch, giống hệt phân phối
    trong training loop, để đảm bảo train_loss và val_loss có thể so sánh
    trực tiếp trên cùng một phân phối QP.
    """
    dmc.eval()

    accum = {
        'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
        'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
    }
    count = 0

    for frames in dataloader:
        frames = frames.to(device)
        T = frames.size(1)
        qp = random.randint(0, 63)  # Cùng phân phối với training

        # I-frame
        x_0 = frames[:, 0, :, :, :]
        x_hat_0, _ = dmci.forward_train(x_0, qp)

        dmc.clear_dpb()
        dmc.set_curr_poc(0)
        dmc.add_ref_frame(feature=None, frame=x_hat_0)

        # P-frames
        for t in range(1, T):
            x_t = frames[:, t, :, :, :]
            x_hat_t, rate_bpp_t = dmc.forward_train(x_t, qp)
            details = criterion(x_t, x_hat_t, rate_bpp_t, return_details=True)

            for k in accum:
                accum[k] += details[k].item()
            count += 1

    for k in accum:
        accum[k] /= max(count, 1)

    dmc.train()
    return accum


@torch.no_grad()
def validate_stage3(dmci, dmc, criterion, dataloader, device, qp_list=None):
    """Validation cho Stage 3 (Joint DMCI+DMC).
    
    QP được bốc ngẫu nhiên từ [0, 63] mỗi batch, giống hệt phân phối
    trong training loop, để đảm bảo train_loss và val_loss có thể so sánh
    trực tiếp trên cùng một phân phối QP.
    """
    dmci.eval()
    dmc.eval()

    accum = {
        'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
        'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
    }
    count = 0

    for frames in dataloader:
        frames = frames.to(device)
        T = frames.size(1)
        qp = random.randint(0, 63)  # Cùng phân phối với training

        # I-frame
        x_0 = frames[:, 0, :, :, :]
        x_hat_0, rate_bpp_0 = dmci.forward_train(x_0, qp)
        details_0 = criterion(x_0, x_hat_0, rate_bpp_0, return_details=True)
        for k in accum:
            accum[k] += details_0[k].item()
        count += 1

        dmc.clear_dpb()
        dmc.set_curr_poc(0)
        dmc.add_ref_frame(feature=None, frame=x_hat_0)

        # P-frames
        for t in range(1, T):
            x_t = frames[:, t, :, :, :]
            x_hat_t, rate_bpp_t = dmc.forward_train(x_t, qp)
            details_t = criterion(x_t, x_hat_t, rate_bpp_t, return_details=True)
            for k in accum:
                accum[k] += details_t[k].item()
            count += 1

    for k in accum:
        accum[k] /= max(count, 1)

    dmci.train()
    dmc.train()
    return accum


# ============================================================================
# Stage 1: Train DMCI (I-frame) với VCM Loss
# ============================================================================


def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
        
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
            
    model.load_state_dict(new_state_dict)
    print(f"  ✓ Loaded checkpoint: {ckpt_path}")
    return ckpt

def train_stage1(args):
    """
    Stage 1: Train DMCI (I-frame codec) với Machine-Oriented Loss.

    Trainable: DMCI + VCMLoss.front_end_trainable
    Frozen: VCMLoss.front_end_original
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  [Stage 1] Train DMCI (I-frame) — VCM Loss")
    print(f"  Device: {device} | Lambda: {args.lambda_base}")
    print(f"{'='*60}\n")

    # ── Dataset ──
    from src.utils.dataset import VimeoSeptupletDataset
    train_dataset = VimeoSeptupletDataset(
        root_dir=args.data_dir, crop_size=args.crop_size,
        num_frames=1,  # Chỉ cần 1 frame cho I-frame
        list_file="sep_trainlist.txt",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = VimeoSeptupletDataset(
            root_dir=args.val_dir, crop_size=args.crop_size,
            num_frames=1, list_file="sep_testlist.txt",
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ── Model ──
    model = DMCI().to(device)
    if args.dmci_ckpt and os.path.exists(args.dmci_ckpt):
        load_checkpoint_weights(model, args.dmci_ckpt, device)

    # ── VCM Loss (Machine-Oriented Objective) ──
    criterion = VCMLoss(
        lambda_base=args.lambda_base,
        model_name=args.task_model,
        extract_layer_idx=args.extract_layer_idx,
    ).to(device)

    # ── Optimizer ──
    # Train cả DMCI + Trainable Front End (theo sơ đồ: gradient chạy qua cả hai)
    trainable_params = list(model.parameters()) + \
                       list(criterion.front_end_trainable.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Logger ──
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'logs'),
        stage=1, lambda_base=args.lambda_base,
    )
    logger.log_config(args)

    best_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Khôi phục trạng thái nếu có resume
    start_epoch = 1
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"  🔄 Đang nạp lại trạng thái (Resume) từ: {args.resume_ckpt}")
        ckpt_data = torch.load(args.resume_ckpt, map_location=device)
        
        # Lấy lại weights
        if 'model_state_dict' in ckpt_data:
            model.load_state_dict(ckpt_data['model_state_dict'])
        if 'front_end_trainable_state_dict' in ckpt_data:
            criterion.front_end_trainable.load_state_dict(ckpt_data['front_end_trainable_state_dict'])
            
        # Lấy lại optimizer và tính toán epoch
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        if 'epoch' in ckpt_data:
            start_epoch = ckpt_data['epoch'] + 1
        
        if 'train_metrics' in ckpt_data and 'total_loss' in ckpt_data['train_metrics']:
            best_loss = ckpt_data['train_metrics']['total_loss']
            
        print(f"  ✅ Sẽ tiếp tục chạy từ Epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        criterion.train()

        epoch_accum = {
            'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
            'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch}/{args.epochs}")
        for batch_idx, frames in enumerate(pbar):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            frames = frames.to(device)
            x = frames[:, 0, :, :, :]  # I-frame chỉ lấy frame đầu

            # Random QP cho multi-quality training
            qp = random.randint(0, 63)

            optimizer.zero_grad()

            # Forward: DMCI → VCM Loss
            x_hat, rate_bpp = model.forward_train(x, qp)
            details = criterion(x, x_hat, rate_bpp, qp=qp, return_details=True)
            loss = details['total_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            for k in epoch_accum:
                epoch_accum[k] += details[k].item()

            # Batch logging
            batch_metrics = {k: details[k].item() for k in epoch_accum}
            logger.log_batch(epoch, batch_idx, batch_metrics)

            if (batch_idx + 1) % 20 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'loss': f'{epoch_accum["total_loss"]/n:.4f}',
                    'bpp': f'{epoch_accum["rate_bpp"]/n:.4f}',
                    'fmse': f'{epoch_accum["feature_mse"]/n:.6f}',
                    'psnr': f'{epoch_accum["pixel_psnr"]/n:.1f}dB',
                })

        scheduler.step()

        # Epoch average
        n_batches = min(batch_idx + 1, len(train_loader))
        train_metrics = {k: v / n_batches for k, v in epoch_accum.items()}

        # Validation
        val_metrics = None
        if val_loader:
            val_metrics = validate_stage1(model, criterion, val_loader, device)
            val_str = (f" | Val Loss={val_metrics['total_loss']:.4f} "
                       f"BPP={val_metrics['rate_bpp']:.4f} "
                       f"PSNR={val_metrics['pixel_psnr']:.1f}dB")
        else:
            val_str = ""

        print(f"  Epoch {epoch}: "
              f"Loss={train_metrics['total_loss']:.4f} | "
              f"BPP={train_metrics['rate_bpp']:.4f} | "
              f"FeatureMSE={train_metrics['feature_mse']:.6f} | "
              f"PSNR={train_metrics['pixel_psnr']:.1f}dB{val_str}")

        # Log epoch
        logger.log_epoch(epoch, scheduler.get_last_lr()[0], train_metrics, val_metrics)

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'front_end_trainable_state_dict': criterion.front_end_trainable.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"vcm_dmci_epoch_{epoch}.pth"))

        # Best model
        check_loss = val_metrics['total_loss'] if val_metrics else train_metrics['total_loss']
        if check_loss < best_loss:
            best_loss = check_loss
            epochs_without_improvement = 0
            torch.save(ckpt, os.path.join(args.save_dir, "vcm_dmci_best.pth"))
            print(f"  ★ Best model saved! (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n  🛑 Early Stopping: Đã {args.patience} epoch liên tiếp không cải thiện. Dừng train sớm.")
            break

    logger.close()
    print(f"\n  Training logs: {logger.epoch_log_path}")
    print(f"  Batch logs: {logger.batch_log_path}")


# ============================================================================
# Stage 2: Train DMC (P-frame) với VCM Loss — DMCI frozen
# ============================================================================

def train_stage2(args):
    """
    Stage 2: Train DMC (P-frame codec) với Machine-Oriented Loss.
    DMCI được đóng băng (frozen), chỉ train DMC + Trainable Front End.

    Luồng mỗi video clip [B, T, 3, H, W]:
      Frame 0: DMCI.forward_train() → I-frame (không gradient)
      Frame 1..T-1: DMC.forward_train() → P-frames (có gradient) → VCM Loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  [Stage 2] Train DMC (P-frame) — VCM Loss, DMCI frozen")
    print(f"  Device: {device} | Lambda: {args.lambda_base}")
    print(f"{'='*60}\n")

    # ── Dataset ──
    from src.utils.dataset import VimeoSeptupletDataset
    train_dataset = VimeoSeptupletDataset(
        root_dir=args.data_dir, crop_size=args.crop_size,
        num_frames=args.num_frames,
        list_file="sep_trainlist.txt",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = VimeoSeptupletDataset(
            root_dir=args.val_dir, crop_size=args.crop_size,
            num_frames=args.num_frames, list_file="sep_testlist.txt",
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ── Models ──
    dmci = DMCI().to(device)
    dmc = DMC().to(device)

    # Load DMCI (bắt buộc)
    if not args.dmci_ckpt or not os.path.exists(args.dmci_ckpt):
        raise FileNotFoundError(
            f"Stage 2 yêu cầu DMCI checkpoint! Không tìm thấy: {args.dmci_ckpt}\n"
            f"Hãy chạy Stage 1 trước hoặc chỉ định --dmci_ckpt"
        )
    dmci_ckpt_data = load_checkpoint_weights(dmci, args.dmci_ckpt, device)

    # Đóng băng DMCI
    dmci.eval()
    for p in dmci.parameters():
        p.requires_grad = False

    # Load DMC (tùy chọn)
    if args.dmc_ckpt and os.path.exists(args.dmc_ckpt):
        load_checkpoint_weights(dmc, args.dmc_ckpt, device)

    # ── VCM Loss ──
    criterion = VCMLoss(
        lambda_base=args.lambda_base,
        model_name=args.task_model,
        extract_layer_idx=args.extract_layer_idx,
    ).to(device)

    # Load trainable front end từ Stage 1 nếu có
    if isinstance(dmci_ckpt_data, dict) and 'front_end_trainable_state_dict' in dmci_ckpt_data:
        criterion.front_end_trainable.load_state_dict(
            dmci_ckpt_data['front_end_trainable_state_dict']
        )
        print(f"  ✓ Loaded Trainable Front End from DMCI checkpoint")
        print(f"  ✓ Loaded Trainable Front End from DMCI checkpoint")

    # ── Optimizer: DMC + Trainable Front End ──
    trainable_params = list(dmc.parameters()) + \
                       list(criterion.front_end_trainable.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Logger ──
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'logs'),
        stage=2, lambda_base=args.lambda_base,
    )
    logger.log_config(args)

    best_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Khôi phục trạng thái nếu có resume
    start_epoch = 1
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"  🔄 Đang nạp lại trạng thái (Resume) từ: {args.resume_ckpt}")
        ckpt_data = torch.load(args.resume_ckpt, map_location=device)
        
        # Lấy lại weights
        if 'model_state_dict' in ckpt_data:
            dmc.load_state_dict(ckpt_data['model_state_dict'])
        if 'front_end_trainable_state_dict' in ckpt_data:
            criterion.front_end_trainable.load_state_dict(ckpt_data['front_end_trainable_state_dict'])
            
        # Lấy lại optimizer và tính toán epoch
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        if 'epoch' in ckpt_data:
            start_epoch = ckpt_data['epoch'] + 1
        
        if 'train_metrics' in ckpt_data and 'total_loss' in ckpt_data['train_metrics']:
            best_loss = ckpt_data['train_metrics']['total_loss']
            
        print(f"  ✅ Sẽ tiếp tục chạy từ Epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        dmc.train()
        criterion.train()

        epoch_accum = {
            'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
            'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch}/{args.epochs}")
        for batch_idx, frames in enumerate(pbar):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            frames = frames.to(device)
            T = frames.size(1)
            base_qp = random.randint(0, 63)

            # Hierarchical QP Offset (bài báo DCVC-RT: [0,8,0,4,0,4,0,4])
            # Cho 5 frame: [0, 4, 0, 4, 0]
            QP_OFFSETS_8 = [0, 8, 0, 4, 0, 4, 0, 4]
            QP_OFFSETS_5 = [0, 4, 0, 4, 0]
            qp_offsets = QP_OFFSETS_5 if T <= 5 else QP_OFFSETS_8

            optimizer.zero_grad()

            # ── Frame 0: I-frame (DMCI, frozen) ──
            x_0 = frames[:, 0, :, :, :]
            qp_0 = min(base_qp + qp_offsets[0], 63)
            with torch.no_grad():
                x_hat_0, _ = dmci.forward_train(x_0, qp_0)

            dmc.clear_dpb()
            dmc.set_curr_poc(0)
            dmc.add_ref_frame(feature=None, frame=x_hat_0)

            # ── Frame 1..T-1: P-frames (DMC, trainable) ──
            batch_loss = torch.tensor(0.0, device=device)
            batch_details = {
                'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
                'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
            }

            for t in range(1, T):
                x_t = frames[:, t, :, :, :]
                qp_t = min(base_qp + qp_offsets[t % len(qp_offsets)], 63)
                x_hat_t, rate_bpp_t = dmc.forward_train(x_t, qp_t)
                details = criterion(x_t, x_hat_t, rate_bpp_t, qp=qp_t, return_details=True)

                batch_loss = batch_loss + details['total_loss']
                for k in batch_details:
                    batch_details[k] += details[k].item()

            # Trung bình qua P-frames
            num_p = T - 1
            batch_loss = batch_loss / num_p

            # Backward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Accumulate
            for k in epoch_accum:
                epoch_accum[k] += batch_details[k] / num_p

            batch_metrics = {k: batch_details[k] / num_p for k in batch_details}
            logger.log_batch(epoch, batch_idx, batch_metrics)

            if (batch_idx + 1) % 20 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'loss': f'{epoch_accum["total_loss"]/n:.4f}',
                    'bpp': f'{epoch_accum["rate_bpp"]/n:.4f}',
                    'fmse': f'{epoch_accum["feature_mse"]/n:.6f}',
                    'psnr': f'{epoch_accum["pixel_psnr"]/n:.1f}dB',
                })

        scheduler.step()

        n_batches = min(batch_idx + 1, len(train_loader))
        train_metrics = {k: v / n_batches for k, v in epoch_accum.items()}

        # Validation
        val_metrics = None
        if val_loader:
            val_metrics = validate_stage2(dmci, dmc, criterion, val_loader, device)
            val_str = (f" | Val Loss={val_metrics['total_loss']:.4f} "
                       f"BPP={val_metrics['rate_bpp']:.4f}")
        else:
            val_str = ""

        print(f"  Epoch {epoch}: "
              f"Loss={train_metrics['total_loss']:.4f} | "
              f"BPP={train_metrics['rate_bpp']:.4f} | "
              f"FeatureMSE={train_metrics['feature_mse']:.6f} | "
              f"PSNR={train_metrics['pixel_psnr']:.1f}dB{val_str}")

        logger.log_epoch(epoch, scheduler.get_last_lr()[0], train_metrics, val_metrics)

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': dmc.state_dict(),
            'front_end_trainable_state_dict': criterion.front_end_trainable.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"vcm_dmc_epoch_{epoch}.pth"))

        check_loss = val_metrics['total_loss'] if val_metrics else train_metrics['total_loss']
        if check_loss < best_loss:
            best_loss = check_loss
            epochs_without_improvement = 0
            torch.save(ckpt, os.path.join(args.save_dir, "vcm_dmc_best.pth"))
            print(f"  ★ Best model saved! (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n  🛑 Early Stopping: Đã {args.patience} epoch liên tiếp không cải thiện. Dừng train sớm.")
            break

    logger.close()
    print(f"\n  Training logs: {logger.epoch_log_path}")


# ============================================================================
# Stage 3: Joint Fine-tune DMCI + DMC với VCM Loss
# ============================================================================

def train_stage3(args):
    """
    Stage 3: Joint fine-tune cả DMCI và DMC cùng lúc với VCM Loss.

    Trainable: DMCI + DMC + Trainable Front End
    Frozen: Original Front End
    Cả I-frame loss và P-frame loss đều tham gia gradient.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  [Stage 3] Joint Fine-tune DMCI + DMC — VCM Loss")
    print(f"  Device: {device} | Lambda: {args.lambda_base}")
    print(f"{'='*60}\n")

    # ── Dataset ──
    from src.utils.dataset import VimeoSeptupletDataset
    train_dataset = VimeoSeptupletDataset(
        root_dir=args.data_dir, crop_size=args.crop_size,
        num_frames=args.num_frames,
        list_file="sep_trainlist.txt",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = VimeoSeptupletDataset(
            root_dir=args.val_dir, crop_size=args.crop_size,
            num_frames=args.num_frames, list_file="sep_testlist.txt",
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ── Models ──
    dmci = DMCI().to(device)
    dmc = DMC().to(device)

    # Load checkpoints
    ckpt_i = load_checkpoint_weights(dmci, args.dmci_ckpt, device) if args.dmci_ckpt and os.path.exists(args.dmci_ckpt) else None
    ckpt_p = load_checkpoint_weights(dmc, args.dmc_ckpt, device) if args.dmc_ckpt and os.path.exists(args.dmc_ckpt) else None

    # ── VCM Loss ──
    criterion = VCMLoss(
        lambda_base=args.lambda_base,
        model_name=args.task_model,
        extract_layer_idx=args.extract_layer_idx,
    ).to(device)

    # Load trainable front end nếu có
    for ckpt_candidate in [
        ckpt_p if args.dmc_ckpt else None,
        ckpt_i if args.dmci_ckpt else None,
    ]:
        if (ckpt_candidate is not None and isinstance(ckpt_candidate, dict)
                and 'front_end_trainable_state_dict' in ckpt_candidate):
            criterion.front_end_trainable.load_state_dict(
                ckpt_candidate['front_end_trainable_state_dict']
            )
            print(f"  ✓ Loaded Trainable Front End from checkpoint")
            break

    # ── Optimizer (learning rate thấp hơn cho fine-tuning) ──
    joint_lr = args.lr * 0.1
    trainable_params = (
        list(dmci.parameters()) +
        list(dmc.parameters()) +
        list(criterion.front_end_trainable.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=joint_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Logger ──
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'logs'),
        stage=3, lambda_base=args.lambda_base,
    )
    logger.log_config(args)

    best_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Khôi phục trạng thái nếu có resume
    start_epoch = 1
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"  🔄 Đang nạp lại trạng thái (Resume) từ: {args.resume_ckpt}")
        ckpt_data = torch.load(args.resume_ckpt, map_location=device)
        
        # Lấy lại weights
        if 'dmci_state_dict' in ckpt_data:
            dmci.load_state_dict(ckpt_data['dmci_state_dict'])
        if 'dmc_state_dict' in ckpt_data:
            dmc.load_state_dict(ckpt_data['dmc_state_dict'])
        if 'front_end_trainable_state_dict' in ckpt_data:
            criterion.front_end_trainable.load_state_dict(ckpt_data['front_end_trainable_state_dict'])
            
        # Lấy lại optimizer và tính toán epoch
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        if 'epoch' in ckpt_data:
            start_epoch = ckpt_data['epoch'] + 1
        
        if 'train_metrics' in ckpt_data and 'total_loss' in ckpt_data['train_metrics']:
            best_loss = ckpt_data['train_metrics']['total_loss']
            
        print(f"  ✅ Sẽ tiếp tục chạy từ Epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        dmci.train()
        dmc.train()
        criterion.train()

        epoch_accum = {
            'total_loss': 0.0, 'rate_bpp': 0.0, 'feature_mse': 0.0,
            'distortion_weighted': 0.0, 'pixel_psnr': 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Stage3 Epoch {epoch}/{args.epochs}")
        for batch_idx, frames in enumerate(pbar):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            frames = frames.to(device)
            T = frames.size(1)
            base_qp = random.randint(0, 63)

            # Hierarchical QP Offset (bài báo DCVC-RT: [0,8,0,4,0,4,0,4])
            QP_OFFSETS_8 = [0, 8, 0, 4, 0, 4, 0, 4]
            QP_OFFSETS_5 = [0, 4, 0, 4, 0]
            qp_offsets = QP_OFFSETS_5 if T <= 5 else QP_OFFSETS_8

            optimizer.zero_grad()

            # ── Frame 0: I-frame (DMCI, có gradient) ──
            x_0 = frames[:, 0, :, :, :]
            qp_0 = min(base_qp + qp_offsets[0], 63)
            x_hat_0, rate_bpp_0 = dmci.forward_train(x_0, qp_0)
            details_0 = criterion(x_0, x_hat_0, rate_bpp_0, qp=qp_0, return_details=True)

            dmc.clear_dpb()
            dmc.set_curr_poc(0)
            dmc.add_ref_frame(feature=None, frame=x_hat_0)

            # Accumulate I-frame
            batch_loss = details_0['total_loss']
            batch_details = {k: details_0[k].item() for k in epoch_accum}

            # ── Frame 1..T-1: P-frames (DMC, có gradient) ──
            for t in range(1, T):
                x_t = frames[:, t, :, :, :]
                qp_t = min(base_qp + qp_offsets[t % len(qp_offsets)], 63)
                x_hat_t, rate_bpp_t = dmc.forward_train(x_t, qp_t)
                details_t = criterion(x_t, x_hat_t, rate_bpp_t, qp=qp_t, return_details=True)

                batch_loss = batch_loss + details_t['total_loss']
                for k in batch_details:
                    batch_details[k] += details_t[k].item()

            # Trung bình qua tất cả frames (I + P)
            batch_loss = batch_loss / T

            # Backward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Accumulate
            for k in epoch_accum:
                epoch_accum[k] += batch_details[k] / T

            batch_metrics = {k: batch_details[k] / T for k in batch_details}
            logger.log_batch(epoch, batch_idx, batch_metrics)

            if (batch_idx + 1) % 20 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'loss': f'{epoch_accum["total_loss"]/n:.4f}',
                    'bpp': f'{epoch_accum["rate_bpp"]/n:.4f}',
                    'fmse': f'{epoch_accum["feature_mse"]/n:.6f}',
                    'psnr': f'{epoch_accum["pixel_psnr"]/n:.1f}dB',
                })

        scheduler.step()

        n_batches = min(batch_idx + 1, len(train_loader))
        train_metrics = {k: v / n_batches for k, v in epoch_accum.items()}

        # Validation
        val_metrics = None
        if val_loader:
            val_metrics = validate_stage3(dmci, dmc, criterion, val_loader, device)
            val_str = (f" | Val Loss={val_metrics['total_loss']:.4f} "
                       f"BPP={val_metrics['rate_bpp']:.4f}")
        else:
            val_str = ""

        print(f"  Epoch {epoch}: "
              f"Loss={train_metrics['total_loss']:.4f} | "
              f"BPP={train_metrics['rate_bpp']:.4f} | "
              f"FeatureMSE={train_metrics['feature_mse']:.6f} | "
              f"PSNR={train_metrics['pixel_psnr']:.1f}dB{val_str}")

        logger.log_epoch(epoch, scheduler.get_last_lr()[0], train_metrics, val_metrics)

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'dmci_state_dict': dmci.state_dict(),
            'dmc_state_dict': dmc.state_dict(),
            'front_end_trainable_state_dict': criterion.front_end_trainable.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"vcm_joint_epoch_{epoch}.pth"))

        check_loss = val_metrics['total_loss'] if val_metrics else train_metrics['total_loss']
        if check_loss < best_loss:
            best_loss = check_loss
            epochs_without_improvement = 0
            torch.save(ckpt, os.path.join(args.save_dir, "vcm_joint_best.pth"))
            # Lưu riêng DMCI và DMC cho tiện sử dụng
            torch.save(dmci.state_dict(),
                       os.path.join(args.save_dir, "vcm_dmci_joint_best.pth"))
            torch.save(dmc.state_dict(),
                       os.path.join(args.save_dir, "vcm_dmc_joint_best.pth"))
            print(f"  ★ Best joint model saved! (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n  🛑 Early Stopping: Đã {args.patience} epoch liên tiếp không cải thiện. Dừng train sớm.")
            break

    logger.close()
    print(f"\n  Training logs: {logger.epoch_log_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DCVC-RT Machine-Oriented Video Codec — Final Training Script"
    )

    # ── Training Stage ──
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Training stage: 1=DMCI, 2=DMC, 3=Joint')

    # ── Data ──
    parser.add_argument('--data_dir', type=str,
                        default='/content/vimeo_septuplet/sequences',
                        help='Đường dẫn tới thư mục sequences của Vimeo90k')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Đường dẫn tới thư mục validation (nếu có)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Kích thước crop (mặc định: 256)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Số frame liên tiếp cho P-frame training (mặc định: 5)')

    # ── Training Hyperparameters ──
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (mặc định: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Số epoch (mặc định: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (mặc định: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Số worker cho DataLoader')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Giới hạn số batch mỗi epoch (cho dry-run/debug)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Số epoch tối đa không cải thiện loss trước khi dừng sớm (Early Stopping). 0 = tắt (mặc định: 5)')

    # ── VCM Loss ──
    parser.add_argument('--lambda_base', type=float, default=256,
                        help='Lambda cho VCM Loss: L = R + λ × MSE(feature) (mặc định: 256)')
    parser.add_argument('--task_model', type=str, default='yolov5s',
                        help='Mô hình task (mặc định: yolov5s)')
    parser.add_argument('--extract_layer_idx', type=int, default=4,
                        help='Layer index để trích feature từ YOLOv5 (mặc định: 4)')

    # ── Checkpoints ──
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Đường dẫn checkpoint để tiếp tục train (khôi phục epoch & optimizer)')
    parser.add_argument('--dmci_ckpt', type=str, default=None,
                        help='Đường dẫn tới DMCI checkpoint')
    parser.add_argument('--dmc_ckpt', type=str, default=None,
                        help='Đường dẫn tới DMC checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints/vcm',
                        help='Thư mục lưu checkpoint (mặc định: checkpoints/vcm)')

    args = parser.parse_args()

    # ── Print Config ──
    print("\n" + "=" * 60)
    print("  DCVC-RT Machine-Oriented Video Codec — Training")
    print("=" * 60)
    print(f"  Stage:        {args.stage}")
    print(f"  Data:         {args.data_dir}")
    print(f"  Validation:   {args.val_dir or 'None (no validation)'}")
    print(f"  Crop size:    {args.crop_size}")
    print(f"  Num frames:   {args.num_frames}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  Lambda:       {args.lambda_base}")
    print(f"  Task model:   {args.task_model}")
    print(f"  Save dir:     {args.save_dir}")
    if args.max_batches:
        print(f"  Max batches:  {args.max_batches} (debug mode)")
    print("=" * 60)

    if args.stage == 1:
        train_stage1(args)
    elif args.stage == 2:
        train_stage2(args)
    elif args.stage == 3:
        train_stage3(args)

    print("\n✅ Training hoàn tất!")


if __name__ == "__main__":
    main()
