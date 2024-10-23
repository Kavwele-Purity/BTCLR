import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

#Data Loading
filepath = '~/Documents/DEEP LEARNING/Breast Tumour Classfication Logistic Regression/DATASETS/WINSCONSIN2/data.csv'
df = pd.read_csv(filepath)

print(df.head())
print(df['diagnosis'].dtype)
df.info()

#Count Distribution Visualization
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
plt.show()

#Data preparation
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
type(df['diagnosis'])
X = df[[col for col in df.columns if col not in ['Unnamed: 32', 'Diagnosis']]]

#X = df.drop(columns = ['Unnamed: 32','diagnosis'], inplace = False)
y = df['diagnosis']
type(df['diagnosis'])

X_samples,X_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1996)

#training set scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32))
y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32))


y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#model training
class LogRegression(nn.Module):

    def __init__(self, input_features):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogRegression(X_features)
#loss and optimization

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

#training epochs
epochs = 100
for epoch in range(epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)%10 == 0:
        print(f"epoch:{epoch+1}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_pred_clasifier = y_predicted.round()
    acc = y_pred_clasifier.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
#DATASET INFORMATION
#MATPLOTLIB
#MODEL TRAININGtorch
#Design model(input, output size, forward pass
# construct loss and optimizer
# Training LoopE
#forward pass
# backward pass
#update weights

