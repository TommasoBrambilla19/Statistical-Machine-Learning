import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.datasets import load_digits
import copy

# ---- Fixed seed ----
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# ---- Load data ----
digits = datasets.load_digits()
X = digits.data              # shape (1797, 64)
y = np.array(digits.target, dtype=int)

# ---- Normalize ----
X = X / 16.0

# ---- Shuffle ----
X, y = shuffle(X, y, random_state=seed)

# ---- Reshape & split ----
N      = 1797       
Ntrain = 1697
Ntest  = 100

X = X[:N]
y = y[:N]

# reshape from (N,64) → (N,1,8,8)
X = X.reshape(N, 1, 8, 8)

Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]

Xtest = X[Ntrain:Ntrain+Ntest]
ytest = y[Ntrain:Ntrain+Ntest]

# ---- Convert to tensors ----
Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.long)

Xtest = torch.tensor(Xtest, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.long)

# ---- Create datasets ----
train_set = TensorDataset(Xtrain, ytrain)
test_set  = TensorDataset(Xtest,  ytest)

# ---- Validation split (keeps test fixed!) ----
val_size = int(0.2 * len(train_set))
train_size = len(train_set) - val_size

training_set, val_set = torch.utils.data.random_split(
    train_set,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)

# ---- DataLoaders ----
train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32)
test_loader  = DataLoader(test_set, batch_size=32)   # ALWAYS identical

# --------------------------
# 2. SIMPLE CNN MODEL (1–2 conv layers)
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (1, 8, 8)
        #2 convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   # → (8, 6, 6)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # → (16, 4, 4)

        # fully connceted neural network
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16*4*4)     # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# --------------------------
# 3. TRAINING LOOP
# --------------------------
epochs = 100
best = 100000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    totalval_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    model.eval()
    for xb, yb in val_loader:
        val_preds = model(xb)
        val_loss = loss_fn(val_preds, yb)

        totalval_loss += val_loss.item()
    # --------- FIXED CODE: DEEP COPY ---------
    if totalval_loss < best:
        best = totalval_loss
        best_model = copy.deepcopy(model.state_dict())
    # -----------------------------------------
    print(f"Epoch {epoch+1}/{epochs} - Best: {best:.6f}")
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")
    print(f"Epoch {epoch+1}/{epochs} - ValLoss: {totalval_loss:.6f}")


# --------------------------
# 4. EVALUATION
# --------------------------
model.load_state_dict(best_model)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        predicted = preds.argmax(dim=1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"\nTest accuracy: {accuracy*100:.2f}%")
