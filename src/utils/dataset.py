import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VimeoSeptupletDataset(Dataset):
    def __init__(self, root_dir, list_file="sep_trainlist.txt", crop_size=256, num_frames=5):
        """
        Dataset loader cho tập Vimeo-90k Septuplet.
        Dành riêng cho việc train trên Google Colab.
        
        Args:
            root_dir: Đường dẫn tới folder chứa ảnh (ví dụ: '/content/vimeo_septuplet/sequences')
            list_file: Tên file text chứa danh sách các folder train (ví dụ: 'sep_trainlist.txt')
            crop_size: Kích thước cắt ngẫu nhiên (bài báo dùng 256x256).
            num_frames: Số lượng frame liên tiếp để train BPTT (bài báo dùng N=5).
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.num_frames = num_frames
        
        # Đọc danh sách các sequence
        list_path = os.path.join(os.path.dirname(root_dir), list_file)
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                self.seq_list = f.read().splitlines()
        else:
            # Fallback nếu không có list_file: lấy tất cả các sub-folder
            self.seq_list = []
            for seq_folder in os.listdir(root_dir):
                seq_path = os.path.join(root_dir, seq_folder)
                if not os.path.isdir(seq_path):
                    continue
                for sub_folder in os.listdir(seq_path):
                    sub_path = os.path.join(seq_path, sub_folder)
                    if not os.path.isdir(sub_path):
                        continue
                    self.seq_list.append(f"{seq_folder}/{sub_folder}")
                    
        # Pipeline biến đổi ảnh:
        # 1. RandomCrop để cắt ảnh thành 256x256
        # 2. ToTensor để chuyển ảnh [0, 255] thành Tensor [0, 1]
        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq_path = os.path.join(self.root_dir, self.seq_list[idx])
        
        # Vimeo-90k septuplet luôn có 7 ảnh (im1.png đến im7.png)
        # Chúng ta cần lấy ngẫu nhiên `num_frames` ảnh liên tiếp (thường là 5)
        start_idx = random.randint(1, 7 - self.num_frames + 1)
        
        frames = []
        # Chú ý: Cần dùng chung một thông số ngẫu nhiên khi RandomCrop cho cả N frame
        # để các frame cắt ra nằm ở cùng một vị trí không gian (spatial location)
        seed = torch.random.initial_seed()
        
        for i in range(self.num_frames):
            img_name = f"im{start_idx + i}.png"
            img_path = os.path.join(seq_path, img_name)
            
            # Load ảnh RGB
            img = Image.open(img_path).convert('RGB')
            
            # Áp dụng chung một seed cho RandomCrop để 5 frame crop cùng 1 góc
            torch.manual_seed(seed)
            img_tensor = self.transform(img)
            frames.append(img_tensor)
            
        # Nối lại thành 1 tensor chung có dạng: [num_frames, 3, 256, 256]
        return torch.stack(frames, dim=0)

# Chú thích: 
# Khi đưa lên Colab, bạn sẽ giải nén Vimeo90k vào /content/vimeo_septuplet
# Gọi dataset bằng: dataset = VimeoSeptupletDataset(root_dir='/content/vimeo_septuplet/sequences')
