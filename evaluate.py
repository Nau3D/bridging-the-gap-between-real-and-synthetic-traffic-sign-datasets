'''
#############################
Scripts used in the making of the paper:

Bridging the Gap Between Real and Synthetic Traffic Sign Repositories
Diogo Lopes da Silva, and António Ramires Fernandes

To appear in proceedings of Delta 2022


#############################
 
Script to evaluate a model on a particular dataset

Usage example:

python scripts/evaluate_no_stn.py --data Belgium/test_images --model Belgium_40.pth --outfile prev_40 --classes 62

This will produce a CSV file with the predictions for every image in the evaluation set

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
import glob

from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from model import Net
import torchvision
import torchvision.transforms as transforms
from math import ceil



parser = argparse.ArgumentParser(description='PyTorch evaluation script')
parser.add_argument('--data', type=str, metavar='D', required=True,
                    help="folder where data is located")
parser.add_argument('--model', type=str, metavar='M', required=True,
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help='name of the output csv file (default: %(default)s)')
parser.add_argument('--classes', type=int, metavar='D', required=True,
                    help="number of classes in model")


args = parser.parse_args()

if args.model==None or args.data == None:
    parser.print_help()
    exit()

number_of_classes = args.classes

state_dict = torch.load(args.model)
model = Net(number_of_classes)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

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


test_dir = args.data

def pil_loader(path):

    with Image.open(path) as img:
        return img.convert('RGB')

transforms = [data_transforms]  
output_file = open(args.outfile + '.csv', "w")

for f in tqdm(os.listdir(test_dir)):
    # if 'ppm' in f:
    output = torch.zeros([1, number_of_classes], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,1): #len(transforms)):
            data = transforms[i](pil_loader(test_dir + '/' + f))
            data = data.cuda()
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            data = Variable(data)
            output = output.cuda()
            output = output.add(model(data))
        pred = output.data.max(1, keepdim=True)[1]
        file_id = f[0:5]
        output_file.write("%s,%s,%d\n" % (file_id, f, pred))



output_file.close()

