'''
#############################
Scripts used in the making of the paper:

Bridging the Gap Between Real and Synthetic Traffic Sign Repositories
Diogo Lopes da Silva, and António Ramires Fernandes

To appear in proceedings of Delta 2022


#############################
 
Training script

We used this script to train our models, both with synthetic and real data.


Usage example:

python train.py --data Belgium/train_gen3_Belgium_templates --seed 0 --runs 1 --epochs 40 



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


import argparse
import glob
import os
import shutil

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Train a number of runs on a dataset')

parser.add_argument('--data', type=str, required = True,
                                        help="folder where original data set is located")
parser.add_argument('--seed', type=int, required=True,
                                        help='first random seed')
parser.add_argument('--runs', type=int, default=1,
                                        help='number of runs (default: %(default)s)')
parser.add_argument('--epochs', type=int, required=True, 
                                        help='total number of epochs per run')
parser.add_argument('--model',  type=str, default='None',
                                        help="Use this option to set a starting model if training fails (default: %(default)s)")
# use hue = 0.2 for croatian datasets, 0.4 otherwise    
parser.add_argument('--hue',  type=float, default=0.4,
                                        help="hue range for dynamic data augmentation to  (default: %(default)s)")
parser.add_argument('--lr_dec',  type=float, default=1,
                                        help="learning rate decrease factor (default: %(default)s)")

args = parser.parse_args()

###################################################
### check if data set exists
if not os.path.exists(args.data):
        exit('data set folder does not exist.')

###################################################
### get number of classes
classes = len ( os.listdir(os.path.join(args.data, 'train_images')))
print('number of classes: ', classes)       

###################################################
### create results folder. exit if it already exists
temp = args.data.split('/')[:-1]
top_folder = os.path.join(*temp)
results_folder = args.data + '/s' + str(args.seed) + '-r' + str(args.runs) + '-c' + str(classes) + '_e' + str(args.epochs) + '_' + str(args.hue) + '_' + str(args.lr_dec)
print('Resuls to be stored in :', results_folder)

if os.path.exists(results_folder):
        exit('results folder already exists.')

os.mkdir(results_folder)

###################################################
### run training procedure
for i in range(args.seed, args.seed + args.runs):
    ### call python script for each run
    if i == args.seed:
        command = 'python scripts/train_dataset.py --data {} --classes {} --seed {}  --epochs {} --model {} --hue {} --lr_dec {}'.format(args.data, classes, i, args.epochs, args.model, args.hue, args.lr_dec)
    else:
        command = 'python scripts/train_dataset.py --data {} --classes {} --seed {}  --epochs {} --hue {} --lr_dec {}'.format(args.data, classes, i, args.epochs, args.hue, args.lr_dec)
    os.system(command)

    ### create folder for run epoch models
    run_folder = os.path.join(results_folder, 'run_{0:02d}'.format(i))
    os.mkdir(run_folder)
    ### copy all epoch models to run folder
    for epoch_file in glob.iglob('*.pth'):
        shutil.move(epoch_file, run_folder)

shutil.move('results.txt', results_folder)
