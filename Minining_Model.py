from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


img_path="/kaggle/input/surface-crack-detection"
dataset=ImageFolder(img_path,transform=ToTensor())
dataset
dataset.classes

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        # the structure of our neural network is as follows:
        # our initial image is sent through a convlutional layer with x number of filters
        # say x=5 filters so our initial input channels are input_i= 3 -> for the rgb channels
        # then we get 5 outputs with the first layer, it is upto us to add padding to preserve the size of the image
        # then we perform a maxpooling operation to reduce the dimensions and hold the important information in the filtered images
        # remember each filter acts as a weight matrix which also adds a bias in the output
        # so essentially a convolutional operation is as such I*F-> element wise multiplication then sum of the multiples similar to
        # w1x1+w2x2+w3x3... after the max pooling operation the resulting matrix/tensor is added by a bias in the background
        # which then gives z^[1]= W^T*X^[1]+B^[1]
        # in the non-linear part we perform usually relu which gives a^[1]= ReLu(z^[1])
        # this is the result of the first convolutional layer which is then sent to a maxpooling layer
        # so we observe that in the convolutional layer we introduce weights and biases.
        # The maxpooling layer does not introduce any weights and biases.
        # in theory the element wise multiplication, addition of bias and non-linear activation fucntion
        # constitute to the convolutional layer.
        # below is the implementation of a simple convolutional neural network that implements
        # 2 convolutional layers , 2 maxpooling layers and 1 fully connected layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=6, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=15, out_channels=10, kernel_size=8, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=6, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=5, kernel_size=5, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(9 * 9 * 5, 50)
        self.linear2 = nn.Linear(50, 10)
        self.linear3 = nn.Linear(10, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        out = out.view(out.size(0), -1)

        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)

        return out

model= CNNmodel()
print(model)
# in this part of the code we first load the data and split it into training and test data
# the data is set to run for 2000 iterations and in each iteration 32 images are sent at once
#the divided data set is sent through a loop where first the
train_set, test_set = torch.utils.data.random_split(dataset, [30000, 10000])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel().to(device)
train_loader = DataLoader(train_set, batch_size=30, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=30, pin_memory=True, num_workers=4)
batch_size = 30
n_iters = 2000
epoches = n_iters / (len(train_set) / batch_size)
epoches = int(epoches)
learning_rate = 0.001
opt = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
gradient = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0
for epoch in range(epoches):
    model.train()
    correct = 0.0
    items = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        yhat = model(X)
        pred = torch.argmax(yhat, 1)

        correct += (y == pred).sum().item()
        items += y.size(0)
        loss = loss_fn(yhat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f'Epoch {epoch} train-loss {loss.item()} train-acc {correct*100/items}')

    # Validation loop
    model.eval()
    correct = 0.0
    items = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            yhat = model(X)
            pred = torch.argmax(yhat, 1)

            correct += (y == pred).sum().item()
            items += y.size(0)
            loss = loss_fn(yhat, y)

        print(f"val-loss {loss.item()} val-acc {correct*100/items}")
predictions=[]
labels=[]
model.eval()
correct=0.0
items=0.0
with torch.no_grad():
    for X, y in test_loader:
        X,y=X.to('cuda'),y.to('cuda')
        yhat=model(X)
        pred=torch.argmax(yhat,1)

        correct+=(y==pred).sum()
        items+=y.size(0)
        loss=loss_fn(yhat,y)
        predictions.extend(pred.cpu())
        labels.extend(y.cpu())

    print(f"Val-acc {correct*100/items}")
confusion_matrix(labels,predictions)
disp=ConfusionMatrixDisplay(confusion_matrix(labels,predictions),display_labels=dataset.classes)
disp.plot()
