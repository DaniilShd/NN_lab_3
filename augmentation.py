from torchvision import transforms
import matplotlib.pyplot as plt
# from data import download_data

train_transforms = transforms.Compose([
    # Преобразуем тензор в PIL Image (если нужно)
    transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy()),  # CHW → HWC и в numpy
    transforms.ToPILImage(),  # numpy array → PIL Image

    # Аугментации
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=36),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    transforms.Resize((180, 180)),

    # Обратно в тензор
    transforms.ToTensor()
])

# Тестовые данные — без аугментации
test_transforms = transforms.Compose([
# Преобразуем тензор в PIL Image (если нужно)
    transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy()),  # CHW → HWC и в numpy
    transforms.ToPILImage(),
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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