import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS 
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# class DownSamplingBlock(nn.Module):
#     def __init__(self, in_filters, out_filters):
#         super(DownSamplingBlock, self).__init__()
#         #TODO rajouter les noms des paramètres
#         self.conv = nn.Conv2d(in_filters, out_filters, 4, 2, 1)
#         self.bn = nn.BatchNorm2d(out_filters)

#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x = self.bn(x)
#         return x

# class UpSamplingBlock(nn.Module):
#     def __init__(self, in_filters, out_filters):
#         super(UpSamplingBlock, self).__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#         #TODO rajouter les noms des paramètres
#         self.conv = nn.Conv2d(in_filters, out_filters, 4, 1, padding=1)
#         self.bn = nn.BatchNorm2d(out_filters)

#     def forward(self, x, skip_input):
#         x = self.up(x)
#         x = F.relu(self.conv(x))
#         x = self.bn(x)
#         return torch.cat([x, skip_imput], dim=0)

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         # Downsampling
#         self.d1 = DownSamplingBlock(1, 32)
#         self.d2 = DownSamplingBlock(32, 64)
#         self.d3 = DownSamplingBlock(64, 128)
#         self.d4 = DownSamplingBlock(128, 256)
#         self.d5 = DownSamplingBlock(256, 256)
#         self.d6 = DownSamplingBlock(256, 256)
#         self.d7 = DownSamplingBlock(256, 256)
                                        
#         # Upsampling
#         self.u1 = UpSamplingBlock(256, 256)
#         self.u2 = UpSamplingBlock(256, 256)
#         self.u3 = UpSamplingBlock(256, 256)
#         self.u4 = UpSamplingBlock(256, 128)
#         self.u5 = UpSamplingBlock(128, 64)
#         self.u6 = UpSamplingBlock(64, 32)
#         #Last Block
#         self.u7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                                 nn.Conv2d(32, 3, 4, 1, padding=1),
#                                 nn.Sigmoid())

#     def forward(self, x):
#         x_d1 = self.d1(x)
#         x_d2 = self.d2(x_d1)
#         x_d3 = self.d3(x_d2)
#         x_d4 = self.d4(x_d3)
#         x_d5 = self.d5(x_d4)
#         x_d6 = self.d6(x_d5)
#         x_d7 = self.d7(x_d6)

#         x_u1 = self.u1(x_d7, x_d6)
#         x_u2 = self.u2(x_u2, x_d5)
#         x_u3 = self.u3(x_u3, x_d4)
#         x_u4 = self.u4(x_u4, x_d3)
#         x_u5 = self.u5(x_u5, x_d2)
#         x_u6 = self.u6(x_u6, x_d1)
#         y = self.d7(x_u6)
        
#         return y

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

    def get_features(self, x):
            conv1 = self.dconv_down1(x)
            x = self.maxpool(conv1)

            conv2 = self.dconv_down2(x)
            x = self.maxpool(conv2)
            
            conv3 = self.dconv_down3(x)
            x = self.maxpool(conv3)
            return x


unet = UNet().cuda()
x = torch.rand((10, 1, 256, 256)).cuda()
print(unet(x).shape)


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

pre_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])])
target_process = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = ImageFolderGrayColor('code/data/landscapes', pre_process, target_process)
loader = DataLoader(dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)

x, y = next(iter(loader))

def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        if writer is not None:
            writer.add_scalar('training loss', mean(running_loss), epoch)
            img_grid = torchvision.utils.make_grid(outputs[:16].detach().cpu())
            writer.add_image('colorized', img_grid, epoch)
            img_grid = torchvision.utils.make_grid(y[:16].detach().cpu())
            writer.add_image('original', img_grid, epoch)

optimizer = optim.Adam(unet.parameters())
writer = SummaryWriter(f'runs/Unet')
writer.add_graph(unet, x.to(device))
train(unet, optimizer, loader, epochs=10, writer=writer)

with torch.no_grad():
    all_embeddings = []
    all_labels = []
    for x, y in loader:
        x , y = x.to(device), y.to(device)
        embeddings = unet.get_features(x).view(-1, 256*28*28)
        all_embeddings.append(embeddings)
        all_labels.append(y)
        if len(all_embeddings)>6:
            break
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    writer.add_embedding(embeddings,
                    label_img=labels, global_step=1)

