import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.utils import shuffle
from sklearn import datasets
import copy
import matplotlib.pyplot as plt


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype=int)

X = X / 16.0
X, y = shuffle(X, y, random_state=seed)

# Use the whole dataset
N      = 1797       
Ntrain = 1697
Ntest  = 00

X = X[:N].reshape(N, 1, 8, 8)
y = y[:N]

Xtrain = X[:Ntrain]; ytrain = y[:Ntrain]
Xtest = X[Ntrain:Ntrain+Ntest]; ytest = y[Ntrain:Ntrain+Ntest]

Xtrain_t = torch.tensor(Xtrain, dtype=torch.float32)
ytrain_t = torch.tensor(ytrain, dtype=torch.long)
Xtest_t  = torch.tensor(Xtest, dtype=torch.float32)
ytest_t  = torch.tensor(ytest, dtype=torch.long)

train_full_set = TensorDataset(Xtrain_t, ytrain_t)
test_set       = TensorDataset(Xtest_t, ytest_t)

val_size = int(0.2 * len(train_full_set))
train_size = len(train_full_set) - val_size

training_set, val_set = random_split(
    train_full_set, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)

train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32)
test_loader  = DataLoader(test_set, batch_size=32)

# Define the Model
class OptimizedCNN(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1)
        #self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels *4 *4, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        #x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def run_training(num_channels):
    model = OptimizedCNN(num_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 30
    best_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(); loss_fn(model(xb), yb).backward(); optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader: val_loss += loss_fn(model(xb), yb).item()
        
        if val_loss < best_loss:
            best_loss = val_loss; best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb); correct += (preds.argmax(dim=1) == yb).sum().item(); total += yb.size(0)
            
    acc = correct / total
    params = sum(p.numel() for p in model.parameters())
    return params, acc, (1-acc)


channels_list = range(2, 26) 
history_params = []
history_err = []

print(f"{'Ch':<4} | {'Params':<8} | {'Error':<8}")
print("-" * 25)

for ch in channels_list:
    p, acc, err = run_training(ch)
    history_params.append(p)
    history_err.append(err)
    print(f"{ch:<4} | {p:<8} | {err:.2%}")


fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Number of Channels (conv1)')
ax1.set_ylabel('Error Rate', color=color)
ax1.plot(channels_list, history_err, 'o-', color=color, label='Error Rate')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Parameter Count', color=color)
ax2.plot(channels_list, history_params, 's--', color=color, label='Model Params')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Trade-off: Error Rate vs Model Complexity')
plt.tight_layout()

print("\nTraining completato. Mostro il grafico finale...")
plt.show()