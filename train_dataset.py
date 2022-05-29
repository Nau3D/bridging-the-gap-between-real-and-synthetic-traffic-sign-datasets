'''
#############################
Scripts used in the making of the paper:

Bridging the Gap Between Real and Synthetic Traffic Sign Repositories
Diogo Lopes da Silva, and António Ramires Fernandes

To appear in proceedings of Delta 2022


#############################
 
Aux script to train a dataset. It is called from train.py

#############################

MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

#############################

'''

####
#### Script adatped from https://github.com/soumith/traffic-sign-detection-homework
####

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import random 
from tqdm import tqdm
import torchvision.transforms as transforms
from math import ceil



# Training settings
parser = argparse.ArgumentParser(description='PyTorch CNN example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: %(default)s)')
parser.add_argument('--classes', type=int, default=43, metavar='D',
                    help="number of classes in model (default: %(default)s)")
parser.add_argument('--model',  type=str, default='None',
                    help="Use this option to set a starting model if training fails (default: %(default)s)")
# use hue = 0.2 for croatian datasets, 0.4 otherwise    
parser.add_argument('--hue',  type=float, default=0.4,
                    help="input image size for model (default: %(default)s)")
parser.add_argument('--lr_dec', type=float, default=1.0, metavar='LR_DEC',
                    help='learning rate decrease (default: %(default)s)')


args = parser.parse_args()


img_size = 32


data_transforms = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    #transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=2),
    transforms.ToTensor()
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(saturation=2),
    #transforms.ColorJitter(saturation=-5),
    transforms.ToTensor()
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(contrast=2),
    #transforms.ColorJitter(contrast=-5),
    transforms.ToTensor()
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(hue=args.hue),
    transforms.ToTensor()
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(5),
    transforms.ToTensor()
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor()
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomAffine(degrees = 5,shear=2),
    transforms.ToTensor()
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.RandomAffine(degrees = 5,translate=(0.1,0.1)),
    transforms.ToTensor()
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
	transforms.Resize((ceil(img_size * 1.1), ceil(img_size * 1.1))),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
	transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

if torch.cuda.is_available():
    use_gpu = True
else:
	use_gpu = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

eval_images_folder = args.data.split('/')[:-1]

eval_images_folder.append('test_images_per_folder')
eval_images_folder = '/'.join(eval_images_folder)

   
# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
   torch.utils.data.ConcatDataset([
       datasets.ImageFolder(args.data + '\\train_images', transform=data_transforms),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_jitter_brightness),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_jitter_hue),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_jitter_contrast),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_jitter_saturation),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_translate),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_rotate),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_center),
       datasets.ImageFolder(args.data + '\\train_images', transform=data_shear)]), 
       batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
   


eval_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(eval_images_folder,
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)
   
#print("Done data loading")

# Neural Network and Optimizer
from model import Net
model = Net(args.classes)




if args.model == 'None':
    first_epoch = 1
else:
    first_epoch = int(args.model.split('_')[-1].split('.')[0]) + 1
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)


if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)

from time import sleep
def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    tk0 = tqdm(train_loader)
    tk0.set_description(f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(tk0):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        
    f = open('results.txt','a')
    f.write('\n{} Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
    f.close()
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
    scheduler.step(training_loss)
    optimizer.param_groups[0]['lr'] *= args.lr_dec

def evaluation():
    model.eval()
    evaluation_loss = 0
    correct = 0
    for data, target in eval_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            evaluation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    evaluation_loss /= len(eval_loader.dataset)
        
    f = open('results.txt','a')
    f.write('Evaluation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        evaluation_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))
    for param_group in optimizer.param_groups:
        f.write(str(param_group['lr']))        
    f.close()
    print('Evaluation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        evaluation_loss, correct, len(eval_loader.dataset),
        correct * 100.0  / len(eval_loader.dataset)))


def main():
    for epoch in range(first_epoch, args.epochs + 1):

        train(epoch)
        evaluation()
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        
if __name__ == '__main__':
    main()
