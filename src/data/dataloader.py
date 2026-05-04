from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.config import DATA_DIR, BATCH_SIZE

def get_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': ImageFolder(f"{DATA_DIR}/train", transform=data_transforms['train']),
        'val': ImageFolder(f"{DATA_DIR}/val", transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False)
    }

    return dataloaders, image_datasets['train'].classes