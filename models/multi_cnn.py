import torch
from datautils import CrimeDataset, EvaluationMetric
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.optim import Adam
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = torch.nn.Linear(131072, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = torch.nn.Linear(625, 5, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.float()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())
np.random.seed(42)


data = CrimeDataset(
    annotations_file="./train_annotation.csv",
    transform = transforms.Resize((256,256))
)

num_train_sample = int(len(data) * 0.8) 
idx = np.random.permutation(list(range(len(data))))
X_train_mask = idx[:num_train_sample] 
X_test_mask = idx[num_train_sample:] 



trainset = torch.utils.data.Subset(data, X_train_mask)
testset = torch.utils.data.Subset(data, X_test_mask)


# dataloader = DataLoader(data, batch_size=32, shuffle=True)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True) # before 64


total_batch = len(trainloader)
print('총 훈련 배치의 수 : {}'.format(total_batch))


learning_rate = 0.001
training_epochs = 30
num_class = 5


model = CNN().to(device) # true 옵션으로 사전 학습된 모델을 로드

optimizer = Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss().to(device)

params = {
    'num_epochs':training_epochs,
    'optimizer':optimizer,
    'loss_function':loss_function,
    'train_dataloader':trainloader,
    'test_dataloader': testloader,
    'device':device
}



def train(model, params):
    num_epochs = params['num_epochs']
    loss_function=params["loss_function"]
    train_dataloader=params["train_dataloader"]
    test_dataloader=params["test_dataloader"]
    device=params["device"]

    for epoch in range(num_epochs):
        model.train()
        for X, Y in tqdm(train_dataloader, desc=f'Epoch {epoch}'):

            X = X.float().to(device)
            Y = Y.to(device)

            optimizer.zero_grad() 

            # forward + back propagation 연산
            outputs = model(X)
            train_loss = loss_function(outputs, Y)
            train_loss.backward()
            optimizer.step()

      # test accuracy 계산

        model.eval()
        correct_y = []
        pred_y = []
        with torch.no_grad():
            for X, Y in tqdm(test_dataloader):
                X = X.float().to(device)
                Y = Y.to(device)

                # 결과값 연산
                outputs = model(X)

                _, predicted = torch.max(outputs.data, 1)
                correct_y.append(Y)
                pred_y.append(predicted)
                test_loss = loss_function(outputs, Y).item()
    
        correct_y = torch.cat(correct_y, axis=0)
        pred_y = torch.cat(pred_y, axis=0)
        correct = len(torch.where(correct_y == pred_y)[0])
        total = len(pred_y)
        test_acc = correct/total

      # 학습 결과 출력
        print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, test_acc))
    
        torch.save(model.state_dict(), './weight_cnn_multi')

    target_names = ['assault','burglary','kidnap','robbery','swoon']
    print(EvaluationMetric(num_class=num_class).multiclass_f_measure(correct_y, pred_y, num_class, target_names))
    a = np.array([correct_y.cpu().numpy(), pred_y.cpu().numpy()])
    np.save('./cnn_test5_predict', a)



s = time.time()

train(model, params)

e = time.time()

print(f'Total Computation Time is {e-s}')



# print(EvaluationMetric(num_class=num_class).multiclass_f_measure(correct_y, pred_y, num_class, target_names))
