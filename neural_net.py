import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CNNBinaryClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)

    self.conv2 = nn.Conv2d(32, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)

    self.conv3 = nn.Conv2d(64, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)

    self.conv4 = nn.Conv2d(128, 256, 3)
    self.bn4 = nn.BatchNorm2d(256)

    self.conv5 = nn.Conv2d(256, 256, 3)
    self.bn5 = nn.BatchNorm2d(256)

    self.pool = nn.MaxPool2d(2, 2)
    # Вычисление размера после всех сверточных слоев
    with torch.no_grad():
      dummy_input = torch.zeros(1, 3, 180, 180)
      x = F.relu(self.bn1(self.conv1(dummy_input)))
      x = self.pool(F.relu(self.bn2(self.conv2(x))))
      x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu(self.bn4(self.conv4(x)))
      x = F.relu(self.bn5(self.conv5(x)))
      x = self.pool(x)
      self.flattened_size = x.view(1, -1).size(1)
      # FC слои
    self.fc = nn.Linear(self.flattened_size, 32)
    self.fc2 = nn.Linear(32, 16)
    self.fc3 = nn.Linear(16, 1)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = F.relu(self.bn5(self.conv5(x)))
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc(x))
    x = F.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


def train(learning_rate, num_epochs, batch_size,
          train_tensor_dataset, test_tensor_dataset):
  # гипер-параметры обучения
  learning_rate = learning_rate  # скорость обучения
  num_epochs = num_epochs  # количество эпох
  batch_size = batch_size  # размер батча

  train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size,
                            shuffle=True)  # shuffle=True

  test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Пример создания модели
  model = CNNBinaryClassifier()
  model = model.to(device)

  criterion = nn.BCELoss()
  optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

  train_losses = []
  val_losses = []

  train_accuracies = []
  val_accuracies = []

  # Основной цикл обучения с валидацией
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Одна эпоха обучения
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)

      # print(outputs.shape)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      print(f"Train Loss: {loss.item()}")
      running_loss += loss.item()
      preds = (outputs > 0.5).float()
      correct_train += (preds == labels).sum().item()
      total_train += labels.size(0)

    # Вычисляем средний трейн-лосс эпохи
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    # Валидация в конце эпохи
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct_val += (preds == labels).sum().item()
        total_val += labels.size(0)

    # Вычисляем средний валидационный лосс эпохи
    val_loss /= len(test_loader)
    val_accuracy = correct_val / total_val
    # Сохраняем метрики
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Эпоха [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    val_accuracy = correct_val / total_val

    # Сохраняем метрики
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Эпоха [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

  test_loss = val_losses[-1]
  test_accuracy = val_accuracies[-1]
  print(f"\nРезультаты на тестовой выборке:\nПотери: {test_loss:.4f}, Точность: {test_accuracy:.4f}")