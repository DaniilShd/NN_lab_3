from torchvision import transforms
import matplotlib.pyplot as plt
# from data import download_data

train_transforms = transforms.Compose([
    transforms.ToPILImage(),  # нужно, если вход — numpy-изображение
    transforms.RandomHorizontalFlip(p=0.5),                # Аналог RandomFlip
    transforms.RandomRotation(degrees=36),                 # Аналог RandomRotation(0.1)
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Аналог RandomZoom(0.2)
    transforms.Resize((180, 180)),                         # Обязательно
    transforms.ToTensor()
])


# Тестовые данные — без аугментации
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((180, 180)),
    transforms.ToTensor()
])


if __name__ == "__main__":
    train_dataset, _ = download_data(f"./data/Train/")
    image_np = train_dataset[7]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        # Применяем аугментацию к NumPy-изображению (HWC, uint8)
        augmented_image = train_transforms(image_np)
        # augmented_image — tensor CxHxW в диапазоне [0,1]
        # Переводим в формат HWC для plt.imshow
        img_to_show = augmented_image.permute(1, 2, 0).numpy()
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img_to_show)
        plt.axis("off")
    plt.show()