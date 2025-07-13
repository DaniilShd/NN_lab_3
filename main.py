import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader
from data import get_datasets, get_aug_datasets
from neural_net import train, train_aug
from torch.utils.data import TensorDataset, DataLoader
import os

if __name__ == "__main__":

    if not os.path.exists("model/model_weights.pth"):
        train_tensor_dataset, test_tensor_dataset = get_datasets("./data")
        train(0.0001, 30, 32,
              train_tensor_dataset, test_tensor_dataset)
    else:
        print("Модель уже обучена")
        # Вариант 1: Только веса (нужно создать экземпляр модели)
        # model = MyNeuralNetwork()  # Сначала создаём модель
        # model.load_state_dict(torch.load('model_weights.pth'))

    if not os.path.exists("model/model_augmentation_weights_base.pth"):
        train_dataset, test_dataset = get_aug_datasets("./data")
        train_aug(0.0001, 30, 32,
              train_dataset, test_dataset)
    else:
        print("Модель уже обучена")
