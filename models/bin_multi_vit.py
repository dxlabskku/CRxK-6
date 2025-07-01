import torch
from datautils import CCTVDataset, EvaluationMetric
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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())


data = CCTVDataset(
    annotations_file="./annotation.csv",
    transform = transforms.Resize((224,224))
)

num_train_sample = int(len(data) * 0.8) 
idx = np.random.permutation(list(range(len(data))))
X_train_mask = idx[:num_train_sample] 
X_test_mask = idx


trainset = torch.utils.data.Subset(data, X_train_mask)
testset = torch.utils.data.Subset(data, X_test_mask)


# dataloader = DataLoader(data, batch_size=32, shuffle=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)


total_batch = len(testloader)
print('총 훈련 배치의 수 : {}'.format(total_batch))


learning_rate = 0.001
training_epochs = 15


model1 = torchvision.models.vit_b_16().to(device) # true 옵션으로 사전 학습된 모델을 로드
model2  = torchvision.models.vit_b_16().to(device)

PATH1 = './weight_vit_bin'
PATH2 = './weight_vit_multi'

model1.load_state_dict(torch.load(PATH1, map_location = 'cuda:3'))
model2.load_state_dict(torch.load(PATH2, map_location = 'cuda:3'))

optimizer1 = Adam(model1.parameters(), lr=learning_rate)
optimizer2 = Adam(model2.parameters(), lr=learning_rate)

loss_function1 = nn.CrossEntropyLoss().to(device)
loss_function2 = nn.CrossEntropyLoss().to(device)

params = {
    'num_epochs':training_epochs,
    'optimizer':optimizer1,
    'loss_function':loss_function1,
    'train_dataloader':trainloader,
    'test_dataloader': testloader,
    'device':device
}


# print(EvaluationMetric.multiclass_f_measure(correct_y, pred_y, num_class, target_names))


def test(model1, model2, params):
    loss_function1=params["loss_function"]
    test_dataloader=params["test_dataloader"]
    device=params["device"]


    model1.eval()
    model2.eval()

    correct_bin_y = []
    pred_bin_y = []

    correct_multi_y = []
    pred_multi_y = []

    correct_all = []

    with torch.no_grad():
        for X, Y in tqdm(test_dataloader):
            X = X.float().to(device)
            Y = Y.to(device)
            Y_bin = torch.where(Y<5, 0, 1)

            # 결과값 연산
            outputs = model1(X)

            _, predicted = torch.max(outputs.data, 1)
            correct_bin_y.append(Y_bin.cpu())
            pred_bin_y.append(predicted.cpu())
            correct_all.append(Y.cpu())



            crimes = (predicted == 0).nonzero(as_tuple=True)[0]
            X = X[crimes]
            Y_multi = Y[crimes]

            outputs = model2(X)

            _, predicted = torch.max(outputs.data, 1)
            correct_multi_y.append(Y_multi.cpu())
            pred_multi_y.append(predicted.cpu())


    correct_all = torch.cat(correct_all, axis=0)

    a = np.array(correct_all.cpu().numpy())
    np.save('./vit_all_label', a)


    correct_bin_y = torch.cat(correct_bin_y, axis=0)
    pred_bin_y = torch.cat(pred_bin_y, axis=0)
    correct = len(torch.where(correct_bin_y == pred_bin_y)[0])
    total = len(pred_bin_y)
    test_acc = correct/total

    target_names1 = ['Crime','Normal']

    print('Binary Test Accuracy at Last Epoch: %2f'%(test_acc))
    print(EvaluationMetric(num_class=2).multiclass_f_measure(correct_bin_y, pred_bin_y, 2, target_names1))

    a = np.array([correct_bin_y.cpu().numpy(), pred_bin_y.cpu().numpy()])
    np.save('./vit_bin5_all', a)


    correct_multi_y = torch.cat(correct_multi_y, axis=0)
    pred_multi_y = torch.cat(pred_multi_y, axis=0)
    correct = len(torch.where(correct_multi_y == pred_multi_y)[0])
    total = len(pred_multi_y)
    test_acc = correct/total


    target_names2 = ['Assault','Burglary','Kidnap','Robbery','Swoon']

    print('Multi Test Accuracy at Last Epoch: %2f'%(test_acc))

    print(EvaluationMetric(num_class=5).multiclass_f_measure(correct_multi_y, pred_multi_y, 5, target_names2))

    a = np.array([correct_multi_y.cpu().numpy(), pred_multi_y.cpu().numpy()])

    np.save('./vit_multi5_all', a)



test(model1,model2, params)
