import argparse
from statistics import mean

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from tqdm import tqdm

from data_utils import get_colorized_dataset_loader
from unet import UNet
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, epochs=5, writer=None):
    criterion = torch.nn.MSELoss()
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
    return mean(running_loss)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = 'data/landscapes', help='dataset path')
    parser.add_argument('--batch_size', type=int, default = int(32), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')

    args = parser.parse_args()
    data_path = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    unet = UNet().cuda()
    loader = get_colorized_dataset_loader(path=data_path, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=4)


    optimizer = optim.Adam(unet.parameters(), lr=lr)
    writer = SummaryWriter('runs/UNet')
    loss = train(unet, optimizer, loader, epochs=10, writer=writer)

    x, y = next(iter(loader))

    with torch.no_grad():
        all_embeddings = []
        all_labels = []
        for x, y in loader:
            x , y = x.to(device), y.to(device)
            embeddings = unet.get_features(x).view(-1, 128*28*28)
            all_embeddings.append(embeddings)
            all_labels.append(y)
            if len(all_embeddings)>6:
                break
        embeddings = torch.cat(all_embeddings)
        labels = torch.cat(all_labels)
        writer.add_embedding(embeddings, label_img=labels, global_step=1)
        writer.add_graph(unet, x.to(device))
        
    # Save model weights
    torch.save(unet.state_dict(), 'unet.pth')
