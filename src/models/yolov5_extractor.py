import torch
import torch.nn as nn

class YOLOv5FeatureExtractor(nn.Module):
    def __init__(self, model_name='yolov5s', extract_layer_idx=4, trainable=False):
        """
        Feature Extractor dựa trên nửa đầu (front-end) của YOLOv5.
        - model_name: Bản YOLOv5 (vd: yolov5s là bản nhỏ nhẹ nhất).
        - extract_layer_idx: Cắt mạng YOLOv5 ở layer số mấy. Bài báo khuyên dùng khoảng layer 4-5.
        - trainable: 
            + False = F_original (đóng băng trọng số, dùng cho ảnh gốc).
            + True = F_trainable (được phép học, dùng cho ảnh giải nén).
        """
        super().__init__()
        # Tự động tải YOLOv5 từ PyTorch Hub (dùng bản v7.0 để tránh lỗi tương thích với thư viện ultralytics mới)
        yolo = torch.hub.load('ultralytics/yolov5:v7.0', model_name, pretrained=True, trust_repo=True)
        
        # Cắt lấy các layer đầu tiên làm "front-end"
        layers = list(yolo.model.model.model.children())[:extract_layer_idx + 1]
        self.feature_extractor = nn.Sequential(*layers)
        
        self.trainable = trainable
        # Nếu đóng băng (F_original), tắt tính toán đạo hàm để tiết kiệm VRAM
        if not trainable:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        # Đầu vào x là Tensor ảnh RGB, giá trị [0, 1]
        if not self.trainable:
            with torch.no_grad():
                return self.feature_extractor(x)
        return self.feature_extractor(x)
