from tkinter import N
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('DATASET.csv')
print(data.head())
print(data.info())
print(data.describe())

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, annot_kws={"size": 12})
plt.show()

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split into features and target
X = scaled_data[:, :-1]  # Features
y = scaled_data[:, -1]   # Target

# Reshape for LSTM input [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Convert to PyTorch tensors
X_torch = torch.from_numpy(X).float()
y_torch = torch.from_numpy(y).float()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.3, shuffle=False)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout_rate=0.3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(1)
        hidden_cell = (torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device),
                       torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device))
        lstm_out, hidden_cell = self.lstm(input_seq, hidden_cell)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Initialize model
input_size = X.shape[2]  # Number of features
hidden_layer_size = 96
output_size = 1
model = LSTM(input_size, hidden_layer_size, output_size)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.068)

# Training loop with early stopping
epochs = 60000
prev_loss = float('inf')
patience = 6000
counter = 0
E = 0
C = 0 

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}')

    # Early stopping
    if epoch_loss < prev_loss:
        prev_loss = epoch_loss
        counter = 0
        C = 0
        E = epoch
        
    else:
        counter += 1
        C += 1
        if counter >= patience:
            print("Early stopping...")
            break
        
        print(f"Epoch = {E}, Min loss = {prev_loss}, Count = {C}\n")

# Testing the model
model.eval()
test_predictions = []

with torch.no_grad():
    for i in range(len(X_test)):
        test_predictions.append(model(X_test[i].unsqueeze(0)).item())

# Inverse scaling
test_predictions = np.array(test_predictions).reshape(-1, 1)
y_test_unscaled = y_test.detach().numpy().reshape(-1, 1)
y_test_unscaled = np.concatenate((X_test.reshape(len(X_test), -1), y_test_unscaled), axis=1)
predicted_values = scaler.inverse_transform(np.concatenate((X_test.reshape(len(X_test), -1), test_predictions), axis=1))
actual_values = scaler.inverse_transform(y_test_unscaled)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(actual_values[:, -1], label='Actual')
plt.plot(predicted_values[:, -1], label='Predicted')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Target')
plt.title('Actual vs Predicted')
plt.show()

# Evaluation metrics
mse = mean_squared_error(actual_values[:, -1], predicted_values[:, -1])
rmse = np.sqrt(mse)
r2 = r2_score(actual_values[:, -1], predicted_values[:, -1])
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")