import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pytorch_lightning as pl

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
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()  # Call plt.show() with parentheses

# Data preparation
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df[[col for col in df.columns if col not in ['Unnamed: 32', 'diagnosis']]]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1996)

# Training set scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32)).view(-1, 1)

# Lightning Module
class LogRegression(pl.LightningModule):
    def __init__(self, input_features):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_features, 1)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred_class = y_pred.round()
        acc = y_pred_class.eq(y).sum() / float(y.shape[0])
        self.log('test_accuracy', acc)  # Log test accuracy

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            batch_size=32, 
            shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test), 
            batch_size=32
        )

# Training
model = LogRegression(X_train.shape[1])

# Trainer Configuration for GPU
trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model)

# Lightning Framework testing
trainer.test(model)  #Test method to evaluate accuracy
