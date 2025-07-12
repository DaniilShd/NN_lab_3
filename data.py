import numpy as np
from PIL import Image, ImageDraw
from skimage import io
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from augmentation import train_transforms, test_transforms


def download_data(path):
  data = []
  y_train = []
  for path_image in sorted(os.listdir(path=path)):
    image = Image.open(path + path_image).resize((180, 180))
    image = np.array(image)
    data.append(image.astype(np.uint8))
    if 'cat' in path_image:
      y_train.append(0)
    else:
      y_train.append(1)
  return np.array(data), np.array(y_train)

def get_datasets(path):
  train_dataset, y_train = download_data(f"{path}/Train/")
  test_dataset, y_test = download_data(f"{path}/Test/")
  # Преобразуем numpy-массивы в тензоры
  # Нормализация + изменение формата: (N, H, W, C) → (N, C, H, W)
  train_tensor = torch.tensor(train_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

  test_tensor = torch.tensor(test_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

  # Создаём датасеты
  train_tensor_dataset = TensorDataset(train_tensor, y_train_tensor)
  test_tensor_dataset = TensorDataset(test_tensor, y_test_tensor)

  return train_tensor_dataset, test_tensor_dataset


class CustomDataset(Dataset):
  def __init__(self, original_dataset, original_y,  transform, num_augmentations=5):
    self.original_dataset = original_dataset
    self.transform = transform
    self.num_augmentations = num_augmentations
    self.original_y = original_y

  def __len__(self):
    return len(self.original_dataset) * (self.num_augmentations + 1)  # +1 для оригинального изображения

  def __getitem__(self, idx):
    original_idx = idx // (self.num_augmentations + 1)
    image = self.original_dataset[original_idx]
    label = self.original_y[original_idx]

    # Для первого элемента возвращаем оригинальное изображение
    if idx % (self.num_augmentations + 1) == 0:
      return image, label

    # Для остальных - аугментированные версии
    return self.transform(image), label

def get_aug_datasets(path):
  train_dataset, y_train = download_data(f"{path}/Train/")
  test_dataset, y_test = download_data(f"{path}/Test/")

  train_tensor = torch.tensor(train_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

  test_tensor = torch.tensor(test_dataset / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


  # Преобразуем numpy-массивы в тензоры
  # Создаем аугментированный датасет (в 5 раз больше)
  augmented_dataset_train = CustomDataset(train_tensor, y_train_tensor,
                                       transform=train_transforms,
                                       num_augmentations=1)

  augmented_dataset_test = CustomDataset(test_tensor, y_test_tensor,
                                          transform=test_transforms,
                                          num_augmentations=0)

  return augmented_dataset_train, augmented_dataset_test

if __name__ == "__main__":

  train_dataset, y_train = download_data(f"./data/Train/")
  test_dataset, y_test = download_data(f"./data/Test/")

  plt.figure(figsize=(8, 8))
  for i in range(4):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_dataset[i])
  plt.figure(figsize=(8, 8))

  for i in range(1200, 1204):
    plt.subplot(4, 4, i - 1199)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_dataset[i])
  plt.show()


