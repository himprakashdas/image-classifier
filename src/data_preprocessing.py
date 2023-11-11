import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
from PIL import Image

class PreprocessedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, img_dir, model_input_shape = (256, 256), transform=None, batch_size=32, shuffle=True) -> None:
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(model_input_shape), # Resizing image
                transform.ToTensor()
            ])

        dataset = datasets.ImageFolder(img_dir, transform=transform)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)