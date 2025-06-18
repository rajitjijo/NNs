import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r"iris.csv")

#NOTE: EDA

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
# fig.tight_layout()

# plots = [(0,1),(2,3),(0,2),(1,3)]
# colors = ['b', 'r', 'g']
# labels = ['Iris setosa','Iris virginica','Iris versicolor']

# for i, ax in enumerate(axes.flat):
#     for j in range(3):
#         x = df.columns[plots[i][0]]
#         y = df.columns[plots[i][1]]
#         ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
#         ax.set(xlabel=x, ylabel=y)

# fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
# plt.show()

features = df.drop("target", axis=1, inplace=False).values
target = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#NOTE: Can be made cleaner with Sequential class

# class IrisModel2(nn.Module):
#     def __init__(self, in_features=4, h1=8, h2=10, out_features=3):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(in_features, h1),
#             nn.ReLU(),
#             nn.Linear(h1, h2),
#             nn.ReLU(),
#             nn.Linear(h2, out_features)
#         )

#     def forward(self, x):
#         return self.network(x)

class IrisModel(nn.Module):
    
    def __init__(self, in_features=4, h1=8, h2=10, out_features=3):
        
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
    
model = IrisModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):

    y_pred = model.forward(X_train)

    loss = criterion(y_pred,y_train)

    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(losses.numpy())

plt.plot(range(epochs), losses)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")
# plt.show()

correct = 0

with torch.no_grad():

    for i, data in enumerate(X_test):

        y_val = model.forward(data)

        print(f"{i}.) {str(y_val)} || {y_test[i]}")

        if y_val.argmax().item() == y_test[i]:
            correct = correct + 1

print(f"We Got {correct} datapoints correct out of {len(X_test)} datapoints")