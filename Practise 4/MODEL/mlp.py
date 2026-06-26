import torch 
import torch.nn as nn

class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceMLP, self).__init__()
        # Định nghĩa kiến trúc mạng (64->32 -> 16 -> 1)
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32,16)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.out(x)
        return x

    def fit(self, train_loader, val_loader, epochs, lr, weight_decay, device):
        self.to(device)
        criterion = nn.MSELoss() # Sử dụng loss MSE
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        print("- start training: ")
        for epoch in range(epochs):
            # --- CHẾ ĐỘ HUẤN LUYỆN (TRAIN) ---
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = self.forward(X_batch)
                
                # Khớp trực tiếp vì y_batch đã được Dataset xử lý log1p sẵn ở trên rồi
                loss = criterion(output, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # --- CHẾ ĐỘ ĐÁNH GIÁ (VALIDATION) ---
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    outputs = self.forward(X_batch)
                    
                    # Khớp trực tiếp, không sử dụng thêm hàm log đè ở đây nữa
                    loss = criterion(outputs, y_batch) 
                    
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            
            # Theo dõi tiến trình qua từng Epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{epochs}] | Train MSE (Log): {train_loss:.4f} | Val MSE (Log): {val_loss:.4f}")
                
        print("-" * 60)
        print("Quá trình huấn luyện hoàn tất!")