import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#use it if you have module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem' error
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x

def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.CrossEntropyLoss()
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

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
  parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')

  args = parser.parse_args()
  exp_name = args.exp_name
  batch_size = args.batch_size
  lr = args.lr

  # setting device on GPU if available, else CPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  net = Net().to(device)
  # default `log_dir` is "runs" - we'll be more specific here
  writer = SummaryWriter(f'runs/{exp_name}')
    
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

  train(net, optimizer, trainloader, 10, writer)
  test_acc = test(net, testloader)
  writer.add_hparams({'lr': lr, 'bsize': batch_size}, {'hparam/accuracy': test_acc}, run_name='MNIST')


  #add embeddings to tensorboard
  perm = torch.randperm(len(trainset.data))
  images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
  images = images.unsqueeze(1).float().to(device)
  with torch.no_grad():
    embeddings = net.get_features(images)
    writer.add_embedding(embeddings,
                  metadata=labels,
                  label_img=images, global_step=1)
    
  # save networks computational graph in tensorboard
  writer.add_graph(net, images)
  # save a dataset sample in tensorboard
  img_grid = torchvision.utils.make_grid(images[:64])
  writer.add_image('mnist_images', img_grid)
