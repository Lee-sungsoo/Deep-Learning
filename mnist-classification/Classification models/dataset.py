import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MNIST(Dataset):
    """
    MNIST dataset
    To write custom datasets, refer to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    
    Args:
        data_dir: directory path containing images
        
    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path) # .convert('L')  # Convert to grayscale
        img = self.transform(img)
        
        label = int(self.image_files[idx].split('_')[1].split('.')[0])
        
        return img, label

if __name__ == '__main__':
    # Test codes to verify implementations
    train_data_dir = '../data/train'
    test_data_dir = '../data/test'
    
    train_dataset = MNIST(train_data_dir)
    test_dataset = MNIST(test_data_dir)
    
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    
    # Get a sample image and label
    img, label = train_dataset[1]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")