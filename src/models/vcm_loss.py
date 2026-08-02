import torch
import torch.nn as nn
from .yolov5_extractor import YOLOv5FeatureExtractor

class VCMLoss(nn.Module):
    def __init__(self, lambda_base=256, model_name='yolov5s', extract_layer_idx=4):
        """
        Hàm Loss Tùy chỉnh (Phương trình 17 trong bài báo "Learned Scalable Video Coding...")
        L_base = R_base + lambda * MSE(r_t, r_hat_t)
        """
        super().__init__()
        self.lambda_base = lambda_base
        
        # F_original: Đóng băng, trích đặc trưng từ video gốc chưa nén
        self.front_end_original = YOLOv5FeatureExtractor(
            model_name=model_name, 
            extract_layer_idx=extract_layer_idx, 
            trainable=False
        )

        
        self.mse_loss = nn.MSELoss()

    def train(self, mode=True):
        """Override train() to ensure F_original always stays in eval mode."""
        super().train(mode)
        self.front_end_original.eval()
        return self

    def forward(self, x_uncompressed, x_reconstructed, rate_bpp, qp=None, return_details=False):
        """
        Đầu vào:
        - x_uncompressed: Ảnh RGB gốc (Shape: [Batch, 3, H, W], Range: [0, 1])
        - x_reconstructed: Ảnh RGB giải nén từ DCVC-RT (Base Frame)
        - rate_bpp: Số bit sinh ra trên mỗi pixel (bpp) được ước tính từ Entropy Model.
        - qp: Quantization parameter index (0-63)
        - return_details: Nếu True, trả về dict chứa tất cả thông tin chi tiết cho logging.
        """
        # 1. Trích xuất đặc trưng (Features)
        # r_t: Feature chuẩn (Ground truth feature)
        r_t = self.front_end_original(x_uncompressed)
        
        # r_hat_t: Feature tái tạo từ ảnh bị nén (Cũng dùng chung Giám khảo Thép)
        r_hat_t = self.front_end_original(x_reconstructed)
        
        # 2. Tính Distortion (Mức độ méo mó của Feature thay vì Pixel)
        feature_mse = self.mse_loss(r_hat_t, r_t)
        
        # 3. Dynamic Lambda Mapping (Gắn nhịp đập Loss với QP)
        if qp is not None:
            import math
            lambda_max = 768.0   # Khớp bài báo DCVC-RT: "interpolated between 1 and 768"
            lambda_min = 1.0
            safe_qp = max(0.0, min(63.0, float(qp)))
            ratio = safe_qp / 63.0
            current_lambda = lambda_max * math.pow(lambda_min / lambda_max, ratio)
        else:
            current_lambda = self.lambda_base
            
        distortion_weighted = current_lambda * feature_mse
        total_loss = rate_bpp + distortion_weighted
        
        if return_details:
            # Tính thêm pixel-level PSNR để giám sát (không ảnh hưởng gradient)
            with torch.no_grad():
                pixel_mse = nn.functional.mse_loss(x_reconstructed, x_uncompressed)
                psnr = 10 * torch.log10(1.0 / pixel_mse.clamp(min=1e-10))
            return {
                'total_loss': total_loss,
                'rate_bpp': rate_bpp,
                'feature_mse': feature_mse,
                'distortion_weighted': distortion_weighted,
                'pixel_psnr': psnr,
            }
        
        return total_loss, feature_mse
