import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_bpp(file_path, num_frames, height, width):
    size_bytes = os.path.getsize(file_path)
    # Total pixels = num_frames * height * width
    bpp = (size_bytes * 8.0) / (num_frames * height * width)
    return bpp

def evaluate_hevc(args):
    print("\n" + "=" * 60)
    print("  HEVC (H.265) Standard Codec Evaluation for VCM")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── YOLOv5 Full Model for Accuracy (Pseudo-Ground Truth) ──
    print(f"  Load Full YOLO Model ({args.task_model}) for Accuracy...")
    try:
        import torchvision
    except ImportError:
        print("  ⚠️ Không tìm thấy torchvision, hãy cài đặt bằng pip install torchvision")
        return
        
    yolo_model = torch.hub.load('ultralytics/yolov5', args.task_model, pretrained=True).to(device)
    yolo_model.eval()

    def compute_yolo_accuracy(preds_orig, preds_comp, iou_thresh=0.5, conf_thresh=0.25):
        p_o = preds_orig[preds_orig[:, 4] > conf_thresh]
        p_c = preds_comp[preds_comp[:, 4] > conf_thresh]
        total_orig = len(p_o)
        total_comp = len(p_c)
        if total_orig == 0:
            return (1.0 if total_comp == 0 else 0.0)
        if total_comp == 0:
            return 0.0
        ious = torchvision.ops.box_iou(p_o[:, :4], p_c[:, :4])
        total_matches = 0
        matched_c = set()
        for i in range(len(p_o)):
            cls_o = p_o[i, 5]
            best_iou = 0
            best_j = -1
            for j in range(len(p_c)):
                if j in matched_c:
                    continue
                if p_c[j, 5] == cls_o and ious[i, j] > iou_thresh:
                    if ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_j = j
            if best_j != -1:
                matched_c.add(best_j)
                total_matches += 1
        precision = total_matches / total_comp if total_comp > 0 else 0.0
        recall = total_matches / total_orig if total_orig > 0 else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ── Dataset ──
    # Dùng chung Dataset loader với VCM để đảm bảo crop và chọn frame giống hệt nhau
    from src.utils.dataset import VimeoSeptupletDataset
    
    # Thiết lập seed cố định để đảm bảo RandomCrop cắt đúng cùng vị trí mỗi lần chạy CRF khác nhau
    torch.manual_seed(42)
    random_seed_for_dataset = 42
    
    test_dataset = VimeoSeptupletDataset(
        root_dir=args.data_dir, crop_size=args.crop_size,
        num_frames=args.num_frames, list_file="sep_testlist.txt",
    )
    
    crf_list = args.crf_list if args.crf_list else [22, 27, 32, 37]

    print(f"  Testing {len(crf_list)} CRF points: {crf_list}")
    print(f"  Dataset: {len(test_dataset)} sequences (Max: {args.max_sequences})\n")

    results = {}

    for crf in crf_list:
        print(f"  ── CRF = {crf} ──")
        
        # Reset seed để đảm bảo CRF nào cũng test trên những frame giống nhau
        import random
        random.seed(random_seed_for_dataset)
        torch.manual_seed(random_seed_for_dataset)

        metrics = {
            'bpp_all': [],
            'feature_mse_all': [],
            'psnr_all': []
        }

        # Tạo thư mục tạm để chứa ảnh cho FFmpeg
        temp_dir = tempfile.mkdtemp()
        
        max_seq = min(args.max_sequences, len(test_dataset)) if args.max_sequences else len(test_dataset)

        try:
            for seq_idx in tqdm(range(max_seq), desc=f"CRF={crf}"):
                frames = test_dataset[seq_idx] # Tensor: [T, 3, H, W], [0, 1]
                T, C, H, W = frames.shape
                
                # Lưu frames ra ảnh PNG để FFmpeg đọc
                input_png_dir = os.path.join(temp_dir, 'input')
                os.makedirs(input_png_dir, exist_ok=True)
                
                from torchvision.utils import save_image
                for t in range(T):
                    save_image(frames[t], os.path.join(input_png_dir, f"im{t+1:03d}.png"))
                
                # Đường dẫn file nén HEVC
                mp4_path = os.path.join(temp_dir, 'output.mp4')
                
                # Gọi FFmpeg nén H.265
                # -preset veryslow để mô phỏng best effort compression
                # -tune zerolatency giúp xử lý video siêu ngắn
                ffmpeg_enc_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-framerate', '30',
                    '-i', os.path.join(input_png_dir, 'im%03d.png'),
                    '-c:v', 'libx265',
                    '-preset', 'veryslow',
                    '-x265-params', f'crf={crf}:bframes=0',
                    '-pix_fmt', 'yuv420p',
                    mp4_path
                ]
                subprocess.run(ffmpeg_enc_cmd, check=True)
                
                # Tính BPP
                bpp = get_bpp(mp4_path, T, H, W)
                
                # Giải nén ra lại thành PNG
                recon_png_dir = os.path.join(temp_dir, 'recon')
                os.makedirs(recon_png_dir, exist_ok=True)
                
                ffmpeg_dec_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', mp4_path,
                    os.path.join(recon_png_dir, 'im%03d.png')
                ]
                subprocess.run(ffmpeg_dec_cmd, check=True)
                
                # Đọc ảnh giải nén vào Tensor
                from PIL import Image
                import torchvision.transforms as transforms
                to_tensor = transforms.ToTensor()
                
                recon_frames = []
                for t in range(T):
                    img = Image.open(os.path.join(recon_png_dir, f"im{t+1:03d}.png")).convert('RGB')
                    recon_frames.append(to_tensor(img))
                recon_frames = torch.stack(recon_frames, dim=0).to(device) # [T, 3, H, W]
                frames = frames.to(device)
                
                # Đánh giá Feature MSE và PSNR cho từng frame
                seq_fmse = []
                seq_psnr = []
                
                with torch.no_grad():
                    for t in range(T):
                        # YOLO Accuracy (F1-score)
                        img_orig = (frames[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img_recon = (recon_frames[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        preds_orig = yolo_model([img_orig]).xyxy[0]
                        preds_recon = yolo_model([img_recon]).xyxy[0]
                        acc = compute_yolo_accuracy(preds_orig, preds_recon)
                        seq_fmse.append(acc)
                        
                        # Pixel PSNR
                        mse = F.mse_loss(recon_frames[t], frames[t]).item()
                        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
                        seq_psnr.append(psnr)
                
                metrics['bpp_all'].append(bpp)
                metrics['feature_mse_all'].append(np.mean(seq_fmse))
                metrics['psnr_all'].append(np.mean(seq_psnr))
                
                # Clean up thư mục input/recon cho lượt lặp tiếp theo
                shutil.rmtree(input_png_dir)
                shutil.rmtree(recon_png_dir)
                os.remove(mp4_path)
                
        finally:
            shutil.rmtree(temp_dir)

        # Aggregate
        results[crf] = {}
        for key, values in metrics.items():
            if values:
                results[crf][f'avg_{key}'] = float(np.mean(values))
                results[crf][f'std_{key}'] = float(np.std(values))

    # ── Print Results Table ──
    print(f"\n{'='*60}")
    print(f"  HEVC Performance Results")
    print(f"{'='*60}")
    print(f"  {'CRF':>4} │ {'BPP (all)':>10} │ {'Accuracy (F1)':>13} │ {'Pixel PSNR':>11}")
    print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*13}─┼─{'─'*11}")

    for crf in sorted(results.keys()):
        r = results[crf]
        bpp_all = r.get('avg_bpp_all', 0)
        fmse = r.get('avg_feature_mse_all', 0)
        psnr = r.get('avg_psnr_all', 0)
        print(f"  {crf:>4} │ {bpp_all:>10.4f} │ {fmse:>13.4f} │ {psnr:>9.2f} dB")
    print(f"{'='*60}\n")

    # ── Save Results ──
    results_path = os.path.join(output_dir, 'anchor_results.json')
    results_json = {str(k): v for k, v in results.items()}
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Kết quả HEVC đã lưu (làm Baseline): {results_path}")
    print(f"  👉 Lệnh so sánh: python evaluate_vcm.py --mode bdrate --anchor_results {results_path} --test_results {os.path.join(output_dir, 'vcm_results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Standard Codec (HEVC) as Anchor")
    parser.add_argument('--data_dir', type=str,
                        default='/content/vimeo_septuplet/sequences',
                        help='Đường dẫn test data')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Kích thước crop')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Số frame cho evaluation')
    parser.add_argument('--crf_list', type=int, nargs='+', default=[22, 27, 32, 37],
                        help='Danh sách CRF để test H.265 (mặc định: 22 27 32 37)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Giới hạn số sequence (cho debug)')
    parser.add_argument('--task_model', type=str, default='yolov5s',
                        help='Mô hình task')
    parser.add_argument('--extract_layer_idx', type=int, default=4,
                        help='Layer index cho feature extraction')
    parser.add_argument('--output_dir', type=str, default='output/evaluation',
                        help='Thư mục lưu kết quả đánh giá')
    args = parser.parse_args()
    
    evaluate_hevc(args)
