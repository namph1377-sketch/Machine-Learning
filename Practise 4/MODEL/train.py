import torch
from torch.utils.data import DataLoader
from data.dataset import KCHouseDataset
from models.mlp import HousePriceMLP
from evaluate.evaluator import evaluate_model

def main():
    # 1. Cấu hình Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 70 # cũ là 100
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Khởi tạo PyTorch DataLoader từ Custom Dataset
    print("Đang tải dữ liệu...")


    train_dataset = KCHouseDataset(file_path=r"C:\Users\This MC\nghi\uth\ml\e4\data\X_train.csv")
    val_dataset   = KCHouseDataset(file_path=r"C:\Users\This MC\nghi\uth\ml\e4\data\X_val.csv")
    test_dataset  = KCHouseDataset(file_path=r"C:\Users\This MC\nghi\uth\ml\e4\data\X_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 3. Khởi tạo Model
    input_dim = train_dataset.X.shape[1] # Lấy số lượng feature (23)
    model = HousePriceMLP(input_dim=input_dim)
    for X, y in train_loader:
        print("min =", y.min().item())
        print("max =", y.max().item())
        break
    # 4. Huấn luyện mô hình (Gọi hàm fit từ class model)
    model.fit(
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        device=DEVICE
    )
    
    # 5. Đánh giá cuối cùng trên tập Test độc lập
    print("\nĐang tiến hành đánh giá trên tập Test...")
# Đường dẫn tới file y_test gốc chứa tiền USD thật chưa log của bạn
    Y_TEST_RAW_PATH = r"C:\Users\This MC\nghi\uth\ml\e4\data\y_test.csv" 
    
    # Gọi hàm đánh giá mới
    evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=DEVICE, 
        y_true_path=Y_TEST_RAW_PATH
    )
    # 6. Lưu trọng số mô hình đã train
    torch.save(model.state_dict(), "best_mlp_model_4layer.pth")
    print("\nĐã lưu model thành công tại 'best_mlp_model.pth'!")

if __name__ == "__main__":
    main()