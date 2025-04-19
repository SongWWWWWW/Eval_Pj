import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    """基于PyTorch实现的多层感知器（MLP）模型，支持CPU和多卡GPU。"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化MLP模型。
        
        参数:
            input_dim (int): 输入特征的维度
            hidden_dim (int): 隐藏层神经元数量
            output_dim (int): 输出维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.scaler = StandardScaler()
        # 设置设备（支持多卡GPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        # 如果有多个GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self = nn.DataParallel(self)
        
    def forward(self, x):
        """
        MLP的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 输出张量
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None, num_epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练MLP模型，并在每个epoch评估测试集，记录最佳RMSE。
        
        参数:
            X_train (np.ndarray): 训练特征
            y_train (np.ndarray): 训练目标
            X_test (np.ndarray, optional): 测试特征
            y_test (np.ndarray, optional): 测试目标
            num_epochs (int): 训练轮数
            batch_size (int): 每个训练批次的大小
            learning_rate (float): 优化器的学习率
            
        返回:
            float: 测试集上的最佳RMSE（如果提供了测试集），否则为None
        """
        # 特征缩放
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        
        # 创建训练数据加载器
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 初始化最佳RMSE和模型状态
        best_rmse = float('inf')
        best_state_dict = None
        
        # 如果提供了测试集，准备测试数据
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=self.device)
        
        # 训练循环
        for epoch in range(num_epochs):
            self.train()  # 设置模型为训练模式
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 在每个epoch评估测试集
            if X_test is not None and y_test is not None:
                self.eval()  # 设置模型为评估模式
                with torch.no_grad():
                    predictions = self.forward(X_test_tensor)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions.cpu().numpy()))
                print(f"Epoch {epoch+1}/{num_epochs}, Test RMSE: {rmse:.4f}")
                
                # 更新最佳RMSE和模型状态
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_state_dict = self.state_dict()
        
        # 恢复最佳模型状态
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
        
        return best_rmse if best_rmse != float('inf') else None
    
    def predict(self, X):
        """
        使用训练好的模型进行预测。
        
        参数:
            X (np.ndarray): 用于预测的输入特征
            
        返回:
            np.ndarray: 预测值
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            predictions = self.forward(X_tensor)
        return predictions.cpu().numpy()  # 返回到CPU以兼容numpy
    
    def evaluate(self, X_test, y_test):
        """
        使用RMSE评估模型性能。
        
        参数:
            X_test (np.ndarray): 测试特征
            y_test (np.ndarray): 测试目标
            
        返回:
            float: 均方根误差
        """
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return rmse

if __name__ == "__main__":
    # 测试 MLP 模型
    X_train = np.random.rand(100, 1)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, 1)
    y_test = np.random.rand(20, 1)
    
    model = MLP(input_dim=1, hidden_dim=64, output_dim=1)
    print(f"Model running on: {model.device}")
    best_rmse = model.train_model(X_train, y_train, X_test, y_test)
    print(f"Best Test RMSE: {best_rmse:.4f}")
    rmse = model.evaluate(X_test, y_test)
    print(f"Final Test RMSE: {rmse:.4f}")