import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import warnings
# warnings.filterwarnings("ignore")

class CCTVDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        

class CrimeDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation = self.annotation[self.annotation['label'] < 5]   ## 5 normal 데이터면 드랍 ##
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    dataset = CrimeDataset(
    annotations_file="./annotation.csv")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0])
    print(type(dataset[0][0]))

class AssaultDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 0, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class BurglaryDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 1, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class KidnapDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 2, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class RobberyDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 3, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class SwoonDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 4, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class NormalDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        self.annotation = pd.read_csv(annotations_file)
        self.annotation = self.annotation.drop(['Unnamed: 0'], axis=1)
        self.annotation.loc[self.annotation['label'] == 5, 'label'] = 10
        self.annotation.loc[self.annotation['label'] != 10, 'label'] = 0
        self.annotation.loc[self.annotation['label'] == 10, 'label'] = 1
        self.img_labels = self.annotation.iloc[:,1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.annotation.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label





class EvaluationMetric:
    def __init__(self, num_class, method='macro'):
        self.num_class = num_class
        self.method = method

    def confusion_matrix(self, num_of_class, pred, true):
        c_m_c = [{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for _ in range(num_of_class)]
        for class_idx in range(num_of_class):
            for idx in range(true.shape[0]):
                if pred[idx] == class_idx:
                    if pred[idx] == true[idx]:
                        c_m_c[class_idx]['TP'] += 1
                    else:
                        c_m_c[class_idx]['FP'] += 1
                else:
                    if true[idx] == class_idx:
                        c_m_c[class_idx]['FN'] += 1
                    else:
                        c_m_c[class_idx]['TN'] += 1
        return c_m_c

    def precision(self, TP, FP, FN, TN):
        if (TP+FP)==0:
            return 0
        else:
            out = TP/(TP+FP)
            return out

    def recall(self, TP, FP, FN, TN):
        if (TP+FN) == 0:
            return 0
        else:
            out = TP/(TP+FN)
            return out

    def f_measure(self, precision, recall, beta=1.0):
        if (precision+recall) == 0:
            return 0
        else:
            out = (beta**2+1)*precision*recall/((beta**2)*precision+recall)
            return out

    def multiclass_f_measure(self, pred, true, num_of_class, target_names):
        f_measure_, precision_, recall_ = [], [], []
        out = {'f_measure': 0.0, 'precision': 0.0, 'recall': 0.0}

        confusion_matrix = self.confusion_matrix(num_of_class, pred, true)

        for class_idx in range(num_of_class):
            confusion_matrix_ = confusion_matrix[class_idx]
            
            precision = self.precision(confusion_matrix_['TP'], confusion_matrix_['FP'], confusion_matrix_['FN'], confusion_matrix_['TN'])
            recall = self.recall(confusion_matrix_['TP'], confusion_matrix_['FP'], confusion_matrix_['FN'], confusion_matrix_['TN'])

            f_measure_.append(self.f_measure(precision, recall))
            precision_.append(precision)
            recall_.append(recall)

        for class_idx in range(num_of_class):
            print(f'For class {target_names[class_idx]}, precision: {precision_[class_idx]:.4f}\trecall: {recall_[class_idx]:.4f}\tf-measure: {f_measure_[class_idx]:.4f}')
        print()
        
        if self.method == 'macro':
            for key, val_per_class in zip(out.keys(), [f_measure_, precision_, recall_]):
                out[key] = sum(val_per_class) / len(val_per_class)
        else:
            raise ValueError('Method should be macro.')

        return out


