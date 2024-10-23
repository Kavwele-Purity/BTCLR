import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data Loading
filepath = '~/Documents/DEEP LEARNING/Breast Tumour Classfication Logistic Regression/DATASETS/WINSCONSIN2/data.csv'
df = pd.read_csv(filepath)

print(df.head())
print(df['diagnosis'].dtype)
df.info()

# Count Distribution Visualization
total = len(df['diagnosis'])
tumour_class = df['diagnosis'].value_counts()

ax = sns.countplot(x=df['diagnosis'], palette="Set2")
plt.title('Count of Malignant (M) vs Benign (B) Diagnoses')
plt.xticks(ticks=[0, 1], labels=['Benign (B)', 'Malignant (M)'])
plt.xlabel('Diagnosis')
plt.ylabel('No. of Cases')
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.show()

# Data preparation
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df[[col for col in df.columns if col not in ['Unnamed: 32', 'diagnosis']]]
y = df['diagnosis']

X_samples, X_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1996)

# Training set scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors and transfer to device
X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)).view(-1, 1).to(device)
y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32)).view(-1, 1).to(device)

# Model definition
class LogRegression(nn.Module):
    def __init__(self, input_features):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

# Initialize model and move to device (GPU)
model = LogRegression(X_features).to(device)

# Loss and optimization
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training epochs
epochs = 100
for epoch in range(epochs):
    model.train()  # Ensure model is in training mode
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()  # set the model is in evaluation mode
with torch.no_grad():
    y_predicted = model(X_test)
    y_pred_classifier = y_predicted.round()
    acc = y_pred_classifier.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc:.4f}')
