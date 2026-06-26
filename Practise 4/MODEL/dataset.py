import pandas as pd
import torch 
from torch.utils.data import Dataset

class KCHouseDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Đường dẫn đến file CSV duy nhất chứa cả features và target 'price_log'.
        """
        # 1. Đọc file CSV duy nhất
        df = pd.read_csv(file_path)
        
        # --- BỔ SUNG: LOẠI BỎ CÁC DÒNG CHỨA NAN ---
        # Kiểm tra xem có dòng nào dính NaN ở bất kỳ cột nào không và xóa hẳn dòng đó đi
# --- BỔ SUNG: LOẠI BỎ CÁC DÒNG CHỨA NAN ---
        # Kiểm tra xem có dòng nào dính NaN ở bất kỳ cột nào không và xóa hẳn dòng đó đi
        if df.isnull().any().any():
            initial_shape = df.shape[0]
            df = df.dropna()
            dropped_rows = initial_shape - df.shape[0]
            
            # SỬA TẠI ĐÂY: Tách việc lấy tên file ra khỏi dấu ngoặc nhọn của f-string
            file_name = file_path.split('\\')[-1]
            print(f"[{file_name}] Đã phát hiện và loại bỏ {dropped_rows} dòng chứa giá trị NaN.")
        
        # 2. Trích xuất target y (cột 'price_log')
        if 'price_log' in df.columns:
            y_values = df['price_log'].values
            self.y = torch.tensor(y_values, dtype=torch.float32).view(-1, 1)
            
            # Xóa cột 'price_log' khỏi DataFrame để nó không bị lẫn vào features X
            df = df.drop(columns=['price_log'])
        else:
            raise KeyError(f"Không tìm thấy cột target 'price_log' trong file {file_path}")
        
        # 3. Loại bỏ thêm cột 'price' gốc nếu lỡ có tồn tại trong file để tránh leak dữ liệu
        if 'price' in df.columns:
            df = df.drop(columns=['price'])
            
        # 4. Giữ lại toàn bộ phần còn lại làm ma trận features X sạch sẽ
        self.X = torch.tensor(df.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]