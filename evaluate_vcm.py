"""
DCVC-RT Machine-Oriented Video Codec — Evaluation Framework
============================================================
Khung đánh giá toàn diện cho mô hình VCM đã train.

3 chế độ đánh giá:
  1. training  — Phân tích loss curves từ training logs (CSV)
  2. vcm       — Đánh giá VCM performance: BPP + Feature MSE + Pixel PSNR
  3. bdrate    — Tính BD-Rate giữa anchor và test
  4. all       — Chạy tất cả

Phần A: Đánh giá Training (Loss Analysis)
  - Vẽ loss curves: Total Loss, Rate (BPP), Feature MSE, PSNR theo epoch
  - Kiểm tra overfitting: so sánh train vs validation loss
  - Phát hiện divergence, instability

Phần B: Đánh giá VCM Performance
  - BPP (Bits Per Pixel) tại nhiều QP
  - Feature MSE (task distortion) tại nhiều QP
  - Pixel PSNR (monitoring metric) tại nhiều QP
  - Bảng tổng hợp kết quả

Phần C: BD-Rate
  - So sánh mô hình VCM vs anchor (DCVC-RT gốc / standard codec)

Sử dụng:
  python evaluate_vcm.py --mode training --log_dir checkpoints/vcm/logs/
  python evaluate_vcm.py --mode vcm --dmci_ckpt ... --dmc_ckpt ... --data_dir ...
  python evaluate_vcm.py --mode bdrate --anchor_results ... --test_results ...
  python evaluate_vcm.py --mode all --log_dir ... --dmci_ckpt ... --dmc_ckpt ...
"""

import os
import csv
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================================
# Phần A: Đánh giá Training — Loss Curves & Overfitting Detection
# ============================================================================

def evaluate_training(args):
    """
    Phân tích training logs (CSV) và tạo báo cáo + biểu đồ.
    """
    print("\n" + "=" * 60)
    print("  [Phần A] Đánh giá Training — Loss Curves")
    print("=" * 60 + "\n")

    log_dir = args.log_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Tìm tất cả epoch log files
    epoch_logs = sorted([
        f for f in os.listdir(log_dir)
        if f.endswith('_epoch.csv')
    ])

    if not epoch_logs:
        print(f"  ✗ Không tìm thấy epoch log files trong {log_dir}")
        return

    print(f"  Tìm thấy {len(epoch_logs)} training runs:\n")

    all_reports = []

    for log_file in epoch_logs:
        log_path = os.path.join(log_dir, log_file)
        print(f"  ── Đang phân tích: {log_file} ──")

        # Đọc CSV
        epochs = []
        train_data = defaultdict(list)
        val_data = defaultdict(list)

        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                for key in ['train_total_loss', 'train_rate_bpp', 'train_feature_mse',
                             'train_distortion_weighted', 'train_pixel_psnr']:
                    if row[key]:
                        train_data[key].append(float(row[key]))

                for key in ['val_total_loss', 'val_rate_bpp', 'val_feature_mse',
                             'val_distortion_weighted', 'val_pixel_psnr']:
                    if row.get(key) and row[key]:
                        val_data[key].append(float(row[key]))

        if not epochs:
            print(f"    ✗ File trống, bỏ qua.")
            continue

        # ── Phân tích ──
        report = {
            'log_file': log_file,
            'num_epochs': len(epochs),
            'final_train_loss': train_data['train_total_loss'][-1],
            'final_train_bpp': train_data['train_rate_bpp'][-1],
            'final_train_feature_mse': train_data['train_feature_mse'][-1],
            'final_train_psnr': train_data['train_pixel_psnr'][-1],
        }

        # Kiểm tra convergence
        if len(train_data['train_total_loss']) >= 3:
            last_3 = train_data['train_total_loss'][-3:]
            loss_change = abs(last_3[-1] - last_3[0]) / max(abs(last_3[0]), 1e-10)
            report['convergence'] = 'Converged' if loss_change < 0.05 else 'Still training'
            report['loss_change_last_3'] = loss_change
        else:
            report['convergence'] = 'Too few epochs'

        # Kiểm tra overfitting (nếu có validation)
        if val_data.get('val_total_loss'):
            report['has_validation'] = True
            report['final_val_loss'] = val_data['val_total_loss'][-1]

            # Overfitting: train loss giảm nhưng val loss tăng
            if len(val_data['val_total_loss']) >= 5:
                train_trend = np.polyfit(range(len(train_data['train_total_loss'][-5:])),
                                         train_data['train_total_loss'][-5:], 1)[0]
                val_trend = np.polyfit(range(len(val_data['val_total_loss'][-5:])),
                                       val_data['val_total_loss'][-5:], 1)[0]

                if train_trend < 0 and val_trend > 0:
                    report['overfitting'] = '⚠️ PHÁT HIỆN OVERFITTING'
                elif val_trend > 0:
                    report['overfitting'] = '⚠️ Có dấu hiệu overfitting nhẹ'
                else:
                    report['overfitting'] = '✓ Không phát hiện overfitting'
            else:
                report['overfitting'] = 'Chưa đủ dữ liệu để đánh giá'
        else:
            report['has_validation'] = False
            report['overfitting'] = 'Không có validation data'

        # Kiểm tra instability (loss tăng đột ngột)
        if len(train_data['train_total_loss']) >= 2:
            losses = np.array(train_data['train_total_loss'])
            diffs = np.diff(losses)
            spikes = np.sum(diffs > np.std(losses) * 2)
            report['instability_spikes'] = int(spikes)
            report['stability'] = '✓ Ổn định' if spikes == 0 else f'⚠️ {spikes} spike(s) phát hiện'
        else:
            report['stability'] = 'Chưa đủ dữ liệu'

        all_reports.append(report)

        # In báo cáo
        print(f"    Epochs:        {report['num_epochs']}")
        print(f"    Final Loss:    {report['final_train_loss']:.4f}")
        print(f"    Final BPP:     {report['final_train_bpp']:.4f}")
        print(f"    Final PSNR:    {report['final_train_psnr']:.1f} dB")
        print(f"    Convergence:   {report['convergence']}")
        print(f"    Stability:     {report['stability']}")
        print(f"    Overfitting:   {report['overfitting']}")
        print()

    # ── Vẽ biểu đồ ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for log_file in epoch_logs:
            log_path = os.path.join(log_dir, log_file)
            epochs = []
            metrics = defaultdict(list)

            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['epoch']))
                    for key in reader.fieldnames:
                        if key != 'epoch' and row[key]:
                            try:
                                metrics[key].append(float(row[key]))
                            except ValueError:
                                pass

            if not epochs:
                continue

            fig_name = log_file.replace('_epoch.csv', '')

            # Plot 1: Loss curves
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Training Analysis — {fig_name}', fontsize=14, fontweight='bold')

            # Total Loss
            ax = axes[0, 0]
            ax.plot(epochs[:len(metrics['train_total_loss'])],
                    metrics['train_total_loss'], 'b-o', label='Train', markersize=3)
            if metrics.get('val_total_loss'):
                ax.plot(epochs[:len(metrics['val_total_loss'])],
                        metrics['val_total_loss'], 'r-s', label='Val', markersize=3)
            ax.set_title('Total Loss (R + λ·D)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rate (BPP)
            ax = axes[0, 1]
            ax.plot(epochs[:len(metrics['train_rate_bpp'])],
                    metrics['train_rate_bpp'], 'b-o', label='Train', markersize=3)
            if metrics.get('val_rate_bpp'):
                ax.plot(epochs[:len(metrics['val_rate_bpp'])],
                        metrics['val_rate_bpp'], 'r-s', label='Val', markersize=3)
            ax.set_title('Rate (BPP)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Bits Per Pixel')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Feature MSE
            ax = axes[1, 0]
            ax.plot(epochs[:len(metrics['train_feature_mse'])],
                    metrics['train_feature_mse'], 'b-o', label='Train', markersize=3)
            if metrics.get('val_feature_mse'):
                ax.plot(epochs[:len(metrics['val_feature_mse'])],
                        metrics['val_feature_mse'], 'r-s', label='Val', markersize=3)
            ax.set_title('Feature MSE (Task Distortion)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Pixel PSNR
            ax = axes[1, 1]
            ax.plot(epochs[:len(metrics['train_pixel_psnr'])],
                    metrics['train_pixel_psnr'], 'b-o', label='Train', markersize=3)
            if metrics.get('val_pixel_psnr'):
                ax.plot(epochs[:len(metrics['val_pixel_psnr'])],
                        metrics['val_pixel_psnr'], 'r-s', label='Val', markersize=3)
            ax.set_title('Pixel PSNR (Monitoring)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{fig_name}_loss_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Biểu đồ đã lưu: {plot_path}")

    except ImportError:
        print("  ℹ️ matplotlib chưa cài đặt, bỏ qua vẽ biểu đồ.")
        print("     Cài đặt: pip install matplotlib")

    # ── Lưu báo cáo JSON ──
    report_path = os.path.join(output_dir, 'training_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Báo cáo training: {report_path}")


# ============================================================================
# Phần B: Đánh giá VCM Performance — BPP + Feature MSE + PSNR
# ============================================================================

def evaluate_vcm_performance(args):
    """
    Đánh giá mô hình VCM trên test set tại nhiều QP.
    Output: bảng BPP, Feature MSE, PSNR cho mỗi QP.
    """
    print("\n" + "=" * 60)
    print("  [Phần B] Đánh giá VCM Performance")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load Models ──
    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.models.vcm_loss import VCMLoss

    dmci = DMCI().to(device)
    dmc = DMC().to(device)

    # Load DMCI
    if args.dmci_ckpt and os.path.exists(args.dmci_ckpt):
        ckpt = torch.load(args.dmci_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            dmci.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'dmci_state_dict' in ckpt:
            dmci.load_state_dict(ckpt['dmci_state_dict'])
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            dmci.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            dmci.load_state_dict(ckpt, strict=False)
        print(f"  ✓ Loaded DMCI: {args.dmci_ckpt}")
    else:
        print(f"  ⚠️ Không tìm thấy DMCI checkpoint, dùng pretrained")
        ckpt_i = torch.load('checkpoints/cvpr2025_image.pth.tar', map_location=device)
        dmci.load_state_dict(ckpt_i)

    # Load DMC
    if args.dmc_ckpt and os.path.exists(args.dmc_ckpt):
        ckpt = torch.load(args.dmc_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            dmc.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and 'dmc_state_dict' in ckpt:
            dmc.load_state_dict(ckpt['dmc_state_dict'])
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            dmc.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            dmc.load_state_dict(ckpt, strict=False)
        print(f"  ✓ Loaded DMC: {args.dmc_ckpt}")
    else:
        print(f"  ⚠️ Không tìm thấy DMC checkpoint, dùng pretrained")
        ckpt_p = torch.load('checkpoints/cvpr2025_video.pth.tar', map_location=device)
        dmc.load_state_dict(ckpt_p)

    dmci.eval()
    dmc.eval()

    # ── YOLOv5 Full Model for mAP (Pseudo-Ground Truth) ──
    print(f"  Load Full YOLO Model ({args.task_model}) for mAP...")
    try:
        import torchvision
    except ImportError:
        print("  ⚠️ Không tìm thấy torchvision, hãy cài đặt bằng pip install torchvision")
        return
        
    # PyTorch 2.6 thay đổi mặc định weights_only=True làm hỏng YOLOv5 v7.0
    # Ta tạm thời vá lỗi (monkey-patch) hàm torch.load để cho phép load toàn bộ model
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    yolo_model = torch.hub.load('ultralytics/yolov5:v7.0', args.task_model, pretrained=True).to(device)
    
    # Trả lại hàm torch.load như cũ
    torch.load = original_load
    yolo_model.eval()

    def compute_mAP(preds_orig, preds_comp, iou_thresh=0.5, conf_thresh=0.25):
        """Compute mAP@0.5 using pseudo-ground-truth (original image detections)."""
        # Ground truth = detections on original (uncompressed) image
        gt_boxes = preds_orig[preds_orig[:, 4] > conf_thresh]
        # Predictions = all detections on compressed image (keep all for PR curve)
        pred_boxes = preds_comp

        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes[pred_boxes[:, 4] > conf_thresh]) == 0 else 0.0

        gt_classes = gt_boxes[:, 5].unique()
        if len(gt_classes) == 0:
            return 1.0

        aps = []
        for cls in gt_classes:
            gt_cls = gt_boxes[gt_boxes[:, 5] == cls]
            pred_cls = pred_boxes[pred_boxes[:, 5] == cls]

            if len(pred_cls) == 0:
                aps.append(0.0)
                continue

            # Sort by confidence descending
            sorted_idx = pred_cls[:, 4].argsort(descending=True)
            pred_cls = pred_cls[sorted_idx]

            ious = torchvision.ops.box_iou(pred_cls[:, :4], gt_cls[:, :4])
            n_gt = len(gt_cls)
            tp = torch.zeros(len(pred_cls))
            fp = torch.zeros(len(pred_cls))
            matched_gt = set()

            for i in range(len(pred_cls)):
                max_iou, max_j = 0, -1
                for j in range(n_gt):
                    if j in matched_gt:
                        continue
                    if ious[i, j] > max_iou:
                        max_iou = ious[i, j].item()
                        max_j = j
                if max_iou >= iou_thresh and max_j != -1:
                    tp[i] = 1
                    matched_gt.add(max_j)
                else:
                    fp[i] = 1

            tp_cum = tp.cumsum(0)
            fp_cum = fp.cumsum(0)
            recall = tp_cum / n_gt
            precision = tp_cum / (tp_cum + fp_cum)

            # All-point interpolation (COCO-style AP)
            recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
            precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
            for i in range(len(precision) - 2, -1, -1):
                precision[i] = max(precision[i], precision[i + 1])
            ap = 0.0
            for i in range(1, len(recall)):
                ap += (recall[i] - recall[i - 1]).item() * precision[i].item()
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0.0

    # ── Dataset ──
    from src.utils.dataset import VimeoSeptupletDataset
    test_dataset = VimeoSeptupletDataset(
        root_dir=args.data_dir, crop_size=args.crop_size,
        num_frames=args.num_frames, list_file="sep_testlist.txt",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── QP List ──
    qp_list = args.qp_list if args.qp_list else [0, 16, 32, 48]

    print(f"  Testing {len(qp_list)} QP points: {qp_list}")
    print(f"  Dataset: {len(test_dataset)} sequences\n")

    # ── Evaluation ──
    results = {}

    for qp in qp_list:
        print(f"  ── QP = {qp} ──")

        metrics = {
            'bpp_i': [], 'bpp_p': [], 'bpp_all': [],
            'feature_mse_i': [], 'feature_mse_p': [], 'feature_mse_all': [],
            'psnr_i': [], 'psnr_p': [], 'psnr_all': [],
        }

        with torch.no_grad():
            for seq_idx, frames in enumerate(tqdm(test_loader, desc=f"QP={qp}")):
                if args.max_sequences and seq_idx >= args.max_sequences:
                    break

                frames = frames.to(device)
                T = frames.size(1)

                # I-frame
                x_0 = frames[:, 0, :, :, :]
                x_hat_0, rate_bpp_0 = dmci.forward_train(x_0, qp)

                # YOLO Accuracy (F1-score)
                img_orig_0 = (x_0.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_recon_0 = (x_hat_0.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                preds_orig_0 = yolo_model([img_orig_0]).xyxy[0]
                preds_recon_0 = yolo_model([img_recon_0]).xyxy[0]
                acc_0 = compute_mAP(preds_orig_0, preds_recon_0)
                # Pixel PSNR
                mse_0 = F.mse_loss(x_hat_0, x_0).item()
                psnr_0 = 10 * np.log10(1.0 / max(mse_0, 1e-10))

                metrics['bpp_i'].append(rate_bpp_0.item())
                metrics['feature_mse_i'].append(acc_0)
                metrics['psnr_i'].append(psnr_0)
                metrics['bpp_all'].append(rate_bpp_0.item())
                metrics['feature_mse_all'].append(acc_0)
                metrics['psnr_all'].append(psnr_0)

                # P-frames
                if T > 1:
                    dmc.clear_dpb()
                    dmc.set_curr_poc(0)
                    dmc.add_ref_frame(feature=None, frame=x_hat_0)

                    for t in range(1, T):
                        x_t = frames[:, t, :, :, :]
                        x_hat_t, rate_bpp_t = dmc.forward_train(x_t, qp)

                        img_orig_t = (x_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img_recon_t = (x_hat_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        preds_orig_t = yolo_model([img_orig_t]).xyxy[0]
                        preds_recon_t = yolo_model([img_recon_t]).xyxy[0]
                        acc_t = compute_mAP(preds_orig_t, preds_recon_t)

                        mse_t = F.mse_loss(x_hat_t, x_t).item()
                        psnr_t = 10 * np.log10(1.0 / max(mse_t, 1e-10))

                        metrics['bpp_p'].append(rate_bpp_t.item())
                        metrics['feature_mse_p'].append(acc_t)
                        metrics['psnr_p'].append(psnr_t)
                        metrics['bpp_all'].append(rate_bpp_t.item())
                        metrics['feature_mse_all'].append(acc_t)
                        metrics['psnr_all'].append(psnr_t)

        # Aggregate
        results[qp] = {}
        for key, values in metrics.items():
            if values:
                results[qp][f'avg_{key}'] = float(np.mean(values))
                results[qp][f'std_{key}'] = float(np.std(values))

    # ── Print Results Table ──
    print(f"\n{'='*80}")
    print(f"  VCM Performance Results")
    print(f"{'='*80}")
    print(f"  {'QP':>4} │ {'BPP (all)':>10} │ {'mAP@0.5':>10} │ {'Pixel PSNR':>11} │ "
          f"{'BPP (I)':>9} │ {'BPP (P)':>9}")
    print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*11}─┼─{'─'*9}─┼─{'─'*9}")

    for qp in sorted(results.keys()):
        r = results[qp]
        bpp_all = r.get('avg_bpp_all', 0)
        fmse = r.get('avg_feature_mse_all', 0)
        psnr = r.get('avg_psnr_all', 0)
        bpp_i = r.get('avg_bpp_i', 0)
        bpp_p = r.get('avg_bpp_p', 0)
        print(f"  {qp:>4} │ {bpp_all:>10.4f} │ {fmse:>10.4f} │ {psnr:>9.2f} dB │ "
              f"{bpp_i:>9.4f} │ {bpp_p:>9.4f}")

    print(f"{'='*80}\n")

    # ── Save Results ──
    results_path = os.path.join(output_dir, 'vcm_results.json')
    # Convert int keys to string for JSON
    results_json = {str(k): v for k, v in results.items()}
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Kết quả đã lưu: {results_path}")

    # ── Plot RD Curves ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        qps = sorted(results.keys())
        bpps = [results[q].get('avg_bpp_all', 0) for q in qps]
        fmses = [results[q].get('avg_feature_mse_all', 0) for q in qps]
        psnrs = [results[q].get('avg_psnr_all', 0) for q in qps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('VCM Rate-Distortion Curves', fontsize=14, fontweight='bold')

        # BPP vs mAP (higher is better)
        ax1.plot(bpps, fmses, 'bo-', label='VCM Model', markersize=8, linewidth=2)
        for i, qp in enumerate(qps):
            ax1.annotate(f'QP={qp}', (bpps[i], fmses[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax1.set_xlabel('BPP (Bits Per Pixel)')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('BPP vs mAP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # BPP vs PSNR
        ax2.plot(bpps, psnrs, 'ro-', label='VCM Model', markersize=8, linewidth=2)
        for i, qp in enumerate(qps):
            ax2.annotate(f'QP={qp}', (bpps[i], psnrs[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax2.set_xlabel('BPP (Bits Per Pixel)')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('BPP vs Pixel PSNR (Monitoring)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'vcm_rd_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ RD curves đã lưu: {plot_path}")

    except ImportError:
        print("  ℹ️ matplotlib chưa cài đặt, bỏ qua vẽ biểu đồ.")

    return results


# ============================================================================
# Phần C: BD-Rate — So sánh giữa anchor và test
# ============================================================================

def evaluate_bdrate(args):
    """
    Tính BD-Rate giữa anchor (DCVC-RT gốc) và test (VCM model).
    """
    print("\n" + "=" * 60)
    print("  [Phần C] BD-Rate Calculation")
    print("=" * 60 + "\n")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    from src.utils.bd_rate import compute_bd_rate, compute_bd_metric

    # Load results
    with open(args.anchor_results, 'r') as f:
        anchor = json.load(f)
    with open(args.test_results, 'r') as f:
        test = json.load(f)

    # Tách QP-indexed results thành arrays
    def extract_rd_points(results):
        """Trích xuất (bpp, feature_mse, psnr) từ JSON results."""
        qps = sorted(results.keys(), key=lambda x: int(x))
        bpps = [results[q]['avg_bpp_all'] for q in qps]
        fmses = [results[q]['avg_feature_mse_all'] for q in qps]
        psnrs = [results[q]['avg_psnr_all'] for q in qps]
        return bpps, fmses, psnrs

    anchor_bpp, anchor_fmse, anchor_psnr = extract_rd_points(anchor)
    test_bpp, test_fmse, test_psnr = extract_rd_points(test)

    print(f"  Anchor: {len(anchor_bpp)} RD points")
    print(f"  Test:   {len(test_bpp)} RD points\n")

    # ── BD-Rate dựa trên YOLO Accuracy (F1-score) ──
    # Lưu ý: Accuracy càng cao càng tốt, nên không cần nghịch đảo.
    anchor_inv_fmse = anchor_fmse
    test_inv_fmse = test_fmse

    results = {}

    try:
        bd_rate_fmse = compute_bd_rate(anchor_bpp, anchor_inv_fmse,
                                        test_bpp, test_inv_fmse)
        bd_metric_fmse = compute_bd_metric(anchor_bpp, anchor_inv_fmse,
                                            test_bpp, test_inv_fmse)
        results['feature_mse'] = {
            'bd_rate_pct': bd_rate_fmse,
            'bd_metric': bd_metric_fmse,
        }
        print(f"  BD-Rate (mAP):  {bd_rate_fmse:+.2f}%")
        print(f"    → {'Test TIẾT KIỆM bitrate' if bd_rate_fmse < 0 else 'Test TỐN THÊM bitrate'}")
        print(f"  BD-Metric (mAP):     {bd_metric_fmse:+.4f}")
    except ValueError as e:
        print(f"  ✗ Không thể tính BD-Rate cho mAP: {e}")
        results['feature_mse'] = {'error': str(e)}

    print()

    # ── BD-Rate dựa trên PSNR (monitoring metric) ──
    try:
        bd_rate_psnr = compute_bd_rate(anchor_bpp, anchor_psnr,
                                        test_bpp, test_psnr)
        bd_psnr = compute_bd_metric(anchor_bpp, anchor_psnr,
                                     test_bpp, test_psnr)
        results['psnr'] = {
            'bd_rate_pct': bd_rate_psnr,
            'bd_psnr_db': bd_psnr,
        }
        print(f"  BD-Rate (Pixel PSNR):   {bd_rate_psnr:+.2f}%")
        print(f"  BD-PSNR:                {bd_psnr:+.2f} dB")
    except ValueError as e:
        print(f"  ✗ Không thể tính BD-Rate cho PSNR: {e}")
        results['psnr'] = {'error': str(e)}

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Tổng kết BD-Rate")
    print(f"{'='*60}")

    if 'bd_rate_pct' in results.get('feature_mse', {}):
        bd_r = results['feature_mse']['bd_rate_pct']
        if bd_r < -5:
            verdict = "✅ VCM model HIỆU QUẢ HƠN anchor đáng kể"
        elif bd_r < 0:
            verdict = "✓ VCM model tốt hơn anchor nhẹ"
        elif bd_r < 5:
            verdict = "≈ Tương đương anchor"
        else:
            verdict = "✗ VCM model kém hơn anchor"
        print(f"  {verdict}")
        print(f"  mAP BD-Rate: {bd_r:+.2f}%")

    # ── Plot comparison ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('BD-Rate Comparison: Anchor vs VCM Model',
                     fontsize=14, fontweight='bold')

        # BPP vs mAP
        ax1.plot(anchor_bpp, anchor_inv_fmse, 'b^-', label='Anchor',
                 markersize=8, linewidth=2)
        ax1.plot(test_bpp, test_inv_fmse, 'ro-', label='VCM Model',
                 markersize=8, linewidth=2)
        ax1.set_xlabel('BPP')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('BPP vs mAP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # BPP vs PSNR
        ax2.plot(anchor_bpp, anchor_psnr, 'b^-', label='Anchor',
                 markersize=8, linewidth=2)
        ax2.plot(test_bpp, test_psnr, 'ro-', label='VCM Model',
                 markersize=8, linewidth=2)
        ax2.set_xlabel('BPP')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('BPP vs Pixel PSNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'bdrate_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Comparison plot: {plot_path}")

    except ImportError:
        pass

    # Save
    results_path = os.path.join(output_dir, 'bdrate_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ BD-Rate results: {results_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DCVC-RT Machine-Oriented Video Codec — Evaluation Framework"
    )

    # ── Mode ──
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'vcm', 'bdrate', 'all'],
                        help='Chế độ đánh giá: training/vcm/bdrate/all')

    # ── Training Analysis ──
    parser.add_argument('--log_dir', type=str, default='checkpoints/vcm/logs',
                        help='Thư mục chứa training logs (CSV)')

    # ── VCM Performance ──
    parser.add_argument('--dmci_ckpt', type=str, default=None,
                        help='Đường dẫn DMCI checkpoint')
    parser.add_argument('--dmc_ckpt', type=str, default=None,
                        help='Đường dẫn DMC checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='/content/vimeo_septuplet/sequences',
                        help='Đường dẫn test data')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Kích thước crop')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Số frame cho evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Số worker cho DataLoader')
    parser.add_argument('--qp_list', type=int, nargs='+', default=None,
                        help='Danh sách QP để test (mặc định: 0 16 32 48)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Giới hạn số sequence (cho debug)')
    parser.add_argument('--task_model', type=str, default='yolov5s',
                        help='Mô hình task')
    parser.add_argument('--extract_layer_idx', type=int, default=4,
                        help='Layer index cho feature extraction')

    # ── BD-Rate ──
    parser.add_argument('--anchor_results', type=str, default=None,
                        help='Đường dẫn file JSON kết quả anchor')
    parser.add_argument('--test_results', type=str, default=None,
                        help='Đường dẫn file JSON kết quả test')

    # ── Output ──
    parser.add_argument('--output_dir', type=str, default='output/evaluation',
                        help='Thư mục lưu kết quả đánh giá')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DCVC-RT Machine-Oriented Video Codec — Evaluation")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    if args.mode in ('training', 'all'):
        evaluate_training(args)

    if args.mode in ('vcm', 'all'):
        evaluate_vcm_performance(args)

    if args.mode in ('bdrate', 'all'):
        if args.anchor_results and args.test_results:
            evaluate_bdrate(args)
        elif args.mode == 'bdrate':
            print("\n  ✗ BD-Rate yêu cầu --anchor_results và --test_results")
        else:
            print("\n  ℹ️ Bỏ qua BD-Rate (cần --anchor_results và --test_results)")

    print("\n✅ Evaluation hoàn tất!")


if __name__ == "__main__":
    main()
