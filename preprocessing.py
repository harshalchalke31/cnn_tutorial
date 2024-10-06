import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class ImagePreProcessing:
    def __init__(self, data_path, target_size=(128, 128), batch_size=32):
        self.data_path = data_path
        self.target_size = target_size
        self.batch_size = batch_size
    def transform_data(self,is_train=True):
        if is_train:
            transform = transforms.Compose([
                transforms.Resize(self.target_size), # Resize the image to target size
                transforms.RandomHorizontalFlip(),      # Flip the image horizontally
                transforms.RandomAffine(scale=(0.8,0.2),shear=0.2,degrees=0),# Shear and zoom augmentation
                transforms.ToTensor()     # Convert to Tensor and rescale pixel values to [0, 1]
            ]) 
            data_path = self.data_path
        else:
            transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor()])
            data_path = self.data_path
        return ImageFolder(root=data_path,transform=transform)

        
