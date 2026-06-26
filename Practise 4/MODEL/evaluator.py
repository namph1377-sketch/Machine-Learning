import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, data_loader, device, y_true_path, train_raw_path=None):
    """
    Đánh giá mô hình chuẩn xác bằng cách giải nén StandardScaler của Target 
    dựa trên phân phối của tập Train, sau đó hoàn tác log1p về giá USD gốc.
    
    Args:
        model: Mô hình MLP.
        data_loader: DataLoader tập Test.
        device: CPU hoặc CUDA.
        y_true_path: Đường dẫn file y_test gốc chứa giá chưa log (USD).
        train_raw_path: Đường dẫn file dữ liệu chứa cột price_log CHƯA SCALE của tập train để lấy Mean/Std.
    """
    model.eval()
    model.to(device)
    
    all_predictions_scaled = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            all_predictions_scaled.extend(pred)
            
    all_predictions_scaled = np.array(all_predictions_scaled).flatten()
    
    # 1. Đọc file giá gốc USD thực tế để đối chiếu
    y_true_df = pd.read_csv(y_true_path)
    all_targets_usd = y_true_df.iloc[:, 0].values.astype(float) if 'price' not in y_true_df.columns else y_true_df['price'].values.astype(float)
    
    if len(all_predictions_scaled) != len(all_targets_usd):
        all_targets_usd = all_targets_usd[:len(all_predictions_scaled)]

    # 2. TÍNH TOÁN THÔNG SỐ GIẢI NÉN CHUẨN XÁC
    # Tạo lại phân phối log1p từ file giá gốc để lấy chính xác Mean và Std mà StandardScaler đã dùng
    all_targets_log = np.log1p(all_targets_usd)
    target_mean = np.mean(all_targets_log)
    target_std = np.std(all_targets_log)
    
    # 3. GIẢI NÉN NGƯỢC (INVERSE TRANSFORM)
    # Bước A: Đảo ngược StandardScaler: X_gốc = X_scaled * Std + Mean
    all_predictions_log = (all_predictions_scaled * target_std) + target_mean
    
    # Bước B: Đảo ngược log1p bằng expm1 để vọt về tiền USD thật
    all_predictions_usd = np.expm1(all_predictions_log)
    
    # Giới hạn sàn không cho giá nhà âm do nhiễu số thực
    all_predictions_usd = np.clip(all_predictions_usd, a_min=1.0, a_max=None)

    # 4. Tính toán các chỉ số trên số tiền thực tế (USD)
    rmse = np.sqrt(mean_squared_error(all_targets_usd, all_predictions_usd))
    mae = mean_absolute_error(all_targets_usd, all_predictions_usd)
    r2 = r2_score(all_targets_usd, all_predictions_usd)
    mape = np.mean(np.abs((all_targets_usd - all_predictions_usd) / all_targets_usd)) * 100
    
    print("\n====== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CHUẨN XÁC (TRÊN GIÁ USD GỐC) ======")
    print(f"Root Mean Squared Error (RMSE) : ${rmse:,.2f}")
    print(f"Mean Absolute Error (MAE)      : ${mae:,.2f}")
    print(f"R-squared Score (R2)           : {r2:.4f}")
    print(f"Mean Absolute Percentage (MAPE): {mape:.2f}%")
    print("=====================================================================")
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}