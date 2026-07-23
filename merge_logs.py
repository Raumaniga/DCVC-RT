import os
import csv
import glob

def merge_csv_logs(log_dir, stage=2):
    print(f"Đang tìm các file log của Stage {stage} trong thư mục: {log_dir}")
    # Tìm tất cả file csv của stage đó
    search_pattern = os.path.join(log_dir, f"stage{stage}_*_epoch.csv")
    csv_files = glob.glob(search_pattern)
    
    # Loại bỏ file merged nếu đã tồn tại trước đó
    merged_filename = f"stage{stage}_merged_epoch.csv"
    merged_path = os.path.join(log_dir, merged_filename)
    if merged_path in csv_files:
        csv_files.remove(merged_path)

    if not csv_files:
        print(f"Không tìm thấy file log nào cho Stage {stage}!")
        return

    print(f"Tìm thấy {len(csv_files)} files. Đang tiến hành gộp...")
    
    # Đọc tất cả các dòng (lưu vào dictionary để nếu trùng epoch sẽ lấy cái sau cùng đè lên)
    merged_data = {}
    fieldnames = None

    # Sắp xếp file theo tên (vì tên chứa timestamp nên sắp xếp tên = sắp xếp thời gian)
    csv_files.sort()

    for file in csv_files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            
            for row in reader:
                epoch = int(row['epoch'])
                merged_data[epoch] = row

    if not merged_data:
        print("Các file csv đều trống!")
        return

    # Ghi ra file mới
    with open(merged_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Ghi theo thứ tự epoch tăng dần
        for epoch in sorted(merged_data.keys()):
            writer.writerow(merged_data[epoch])

    print(f"\n✅ Đã gộp thành công toàn bộ các mảnh epoch từ 1 đến {max(merged_data.keys())}!")
    print(f"File tổng hợp được lưu tại: {merged_path}")
    print("\n👉 BÂY GIỜ HÃY XÓA/DI CHUYỂN CÁC FILE CSV CŨ đi, chỉ để lại file 'merged' này và chạy evaluate_vcm.py nhé!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/model_moi/checkpoints/logs')
    parser.add_argument('--stage', type=int, default=2)
    args = parser.parse_args()
    
    merge_csv_logs(args.log_dir, args.stage)
