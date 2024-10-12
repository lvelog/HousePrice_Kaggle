import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device('cuda')
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

train_data = pd.read_csv('dataset\\train_process.csv')
test_data = pd.read_csv('dataset\\test_process.csv')
train_label = train_data[['SalePrice']]
train_data = train_data.drop(columns=['SalePrice'])

min_price = train_label['SalePrice'].min()
max_price = train_label['SalePrice'].max()

numeric_features_label = train_label.dtypes[train_label.dtypes != 'object'].index
train_label.loc[:,numeric_features_label] = train_label[numeric_features_label].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

combined_data = pd.concat([train_data, test_data], keys=['train', 'test'])
numeric_features_train = combined_data.dtypes[combined_data.dtypes != 'object'].index
combined_data[numeric_features_train] = combined_data[numeric_features_train].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())) # 对数值数据变为总体为均值为0，方差为1的分布的数据
combined_data = pd.get_dummies(combined_data,dummy_na=True)  # 处理离散值。用一次独热编码替换它们
combined_data[numeric_features_train] = combined_data[numeric_features_train].fillna(0)  # 将数值数据中not number的数据用0填充
train_data = combined_data.xs('train')
test_data = combined_data.xs('test')


train_features = torch.tensor(train_data.values.astype(float),dtype=torch.float32).to(device)
test_features = torch.tensor(test_data.values.astype(float),dtype=torch.float32).to(device)
train_labels = torch.tensor(train_label.values.astype(float).reshape(-1,1),dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 随机洗牌数据
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 验证集不需要洗牌


model = nn.Sequential(
    nn.Linear(64,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1),
).to(device)

train_losses = []
test_losses = []

best_test_loss = float('inf')
best_model_path = 'best_model_weights.pth'

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10000

mode = 'predict'
if mode == 'train':
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)  # 将批次数据移动到GPU
            y_pred = model(train_features)
            train_loss = criterion(y_pred, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
            test_losses.append(test_loss.item())
        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], train_Loss: {train_loss.item()}, test_Loss: {test_loss.item()}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型权重
    plot_losses(train_losses,test_losses)

elif mode == 'predict':
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(test_features).detach().cpu().numpy()
        predicted_sale_prices = predicted_normalized * (max_price - min_price) + min_price
        predicted_sale_prices_rounded = np.round(predicted_sale_prices, decimals=4)
        predicted_sale_prices_formatted = [f"{price.item():.4f}" for price in predicted_sale_prices_rounded]

        df_predictions = pd.DataFrame(predicted_sale_prices_formatted, columns=["SalePrice"])

        df_predictions.index = range(1461, 1461 + len(df_predictions))
        df_predictions.rename_axis('Id', axis=0, inplace=True)
        df_predictions.to_csv("predicted_sale_prices.csv", index=True)