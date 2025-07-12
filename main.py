import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader
from data import get_datasets
from neural_net import train
from torch.utils.data import TensorDataset, DataLoader

if __name__ == "__main__":
  train_tensor_dataset, test_tensor_dataset = get_datasets("./data")

  train(0.0001, 31, 32,
          train_tensor_dataset, test_tensor_dataset)