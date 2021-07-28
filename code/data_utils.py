from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class ImageFolderGrayColor(ImageFolder):
    
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
    ):
        super(ImageFolder, self).__init__(root=root,
                                          loader=default_loader,
                                          transform=transform,
                                          extensions=IMG_EXTENSIONS,
                                          target_transform=target_transform)

    #TODO à modifier
    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, _ = self.samples[index]
            sample = self.loader(path)
            if self.target_transform is not None:
                target = self.target_transform(sample)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target


def get_colorized_dataset_loader(path, **kwargs):
    source_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])
    target_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolderGrayColor(path, source_process, target_process)
    return DataLoader(dataset, **kwargs)