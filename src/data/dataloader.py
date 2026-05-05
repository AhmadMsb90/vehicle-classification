from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import IMG_SIZE, BATCH_SIZE

# returns image transforms for training or validation
def get_transforms(train=True):
    if train:
        # training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        # validation transforms without augmentation
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# function loads train and validation datasets and returns dataloaders
def get_dataloaders(data_dir):
    # load training dataset
    train_ds = datasets.ImageFolder(
        f"{data_dir}/train",
        transform=get_transforms(True)
    )

    # load validation dataset
    val_ds = datasets.ImageFolder(
        f"{data_dir}/val",
        transform=get_transforms(False)
    )

    # create training dataloader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # create validation dataloader
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # return loaders and class names
    return train_loader, val_loader, train_ds.classes
