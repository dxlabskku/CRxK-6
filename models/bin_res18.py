import torch
from datautils import NormalDataset, EvaluationMetric
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


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

data = NormalDataset(
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
testloader = DataLoader(testset, batch_size=128, shuffle=True)


total_batch = len(trainloader)
print('총 훈련 배치의 수 : {}'.format(total_batch))

category = 'Normal Dataset'

learning_rate = 0.001
training_epochs = 45
num_class = 2

model = torchvision.models.resnet18(pretrained=True).to(device) # true 옵션으로 사전 학습된 모델을 로드

# PATH = './weight_bin_normal'
# model.load_state_dict(torch.load(PATH))

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
        for X, Y in tqdm(train_dataloader, desc=f'Epoch{epoch}'):

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

        print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, test_acc))
        torch.save(model.state_dict(), './weight5_bin_normal_new')

    target_names = ['Crime','Normal']
    print(EvaluationMetric(num_class=num_class).multiclass_f_measure(correct_y, pred_y, num_class, target_names))

    a = np.array([correct_y.cpu().numpy(), pred_y.cpu().numpy()])

    np.save('./res18_binary_test5_new', a)


s = time.time()

train(model, params)

e = time.time()

print(f'Total Computation Time is {e-s}')