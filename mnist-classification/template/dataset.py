import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform

        # get the list of image filenames
        self.filenames = os.listdir(data_dir)

    def __len__(self):

        
        return len(self.filenames)

    def __getitem__(self, idx):

        # write your codes here
        
        # get the image filename
        filename = self.filenames[idx]

        # load image
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path).convert('L') # Convert to grayscale
        
        if self.transform is not None:
            img = self.transform(img)

        # Extract the label from the filename
        label = int(filename.split('_')[1].split('.')[0])

        return img, label

if __name__ == '__main__':

    # data directory
    train_data_dir = '../data/train'
    test_data_dir = '../data/test'

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create instances of the MNIST dataset for train and test splits
    train_dataset = MNIST(train_data_dir, transform=transform)
    test_dataset = MNIST(test_data_dir, transform=transform)

    # # Define the match size
    # batch_size = 32

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    