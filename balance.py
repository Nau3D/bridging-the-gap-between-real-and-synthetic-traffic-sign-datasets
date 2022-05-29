'''
#############################
Scripts used in the making of the paper:

Bridging the Gap Between Real and Synthetic Traffic Sign Repositories
Diogo Lopes da Silva, and António Ramires Fernandes

To appear in proceedings of Delta 2022


#############################
 
Script to balance datasets

The dataset is augmented such that each class has at least "min" elements. Vertical flips are used primary, taking into account that the flipped sample may belong to another class

Usage example:

python balance.py --data original_data_location --seed 0 -- min 2000 --flips german

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

from PIL import Image
import PIL.ImageOps    
import os
import shutil
import random
from random import *
from tqdm import tqdm
import re

import argparse

parser = argparse.ArgumentParser(description='Create a balanced data set')
parser.add_argument('--data', type=str, required = True,
                    help="folder where original data set is located")
parser.add_argument('--seed', type=int, default=0, 
                    help='random seed (default: %(default)s)')
parser.add_argument('--min', type=int, default=2000, 
                    help='minimum number of images per class (default: %(default)s)')

parser.add_argument('--flips', choices=['belgium', 'croatian', 'german', 'none'], required=True, 
                    help='country is required to use flipping')

args = parser.parse_args()

seed(args.seed)    

################################################
### define flips for know country datasets

if args.flips == 'belgium':
    flippable = {'00000': '00000',
			'00001': '00001',
			'00003': '00004',
			'00004': '00003',
			'00006': '00005',
			'00005': '00006', 
			'00011': '00011',
			'00012': '00012',
			'00013': '00013',
			'00014': '00014',
			'00015': '00016',
			'00016': '00015',
			'00017': '00017',
			'00018': '00018',
			'00019': '00019',
			'00022': '00022',
			'00028': '00028',
			'00034': '00034',
			'00035': '00035',
			'00041': '00041',
			'00053': '00053',
			'00054': '00054',
			'00059': '00059',
			'00061': '00061',
			}
elif args.flips ==  'croatian':
    flippable = {'00000': '00000',
			'00001': '00001',
			'00002': '00003',
			'00003': '00002',
			'00004': '00005',
			'00005': '00004', 
			'00006': '00007',
			'00007': '00006',
			'00008': '00008',
			'00014': '00014',
			'00016': '00016',
			'00023': '00023',
			}
elif args.flips == 'german':
    flippable = {'00011': '00011',
			'00012': '00012',
			'00013': '00013',
			'00015': '00015',
			'00017': '00017',
			'00018': '00018', 
			'00019': '00020',
			'00020': '00019',
			'00022': '00022',
			'00026': '00026',
			'00030': '00030',
			'00033': '00034',
			'00034': '00033',
			'00035': '00035',
            '00036': '00037',
			'00037': '00036',
			'00038': '00039',
			'00039': '00038',
			}
else:
    flippable = {} 


################################################

output_folder = re.split('/',args.data)[-1]
top_folder = re.split('/',args.data)[:-1]
top_folder = '/'.join(top_folder)

################################################
### check if data folder exists

if not os.path.exists(args.data):
        exit('data set folder does not exist.')

################################################
### output folder will have a name to allow to identify the original set

dest_folder = output_folder + '_balanced_s' + str(args.seed) + '_m' + str(args.min) + '_' + args.flips
if not top_folder == '':    
    dest_folder = top_folder + '/' + dest_folder

print('destination folder: ', dest_folder)
if os.path.exists(dest_folder):
        exit('output folder already exists.')


################################################
### copy the original data set
print('copying data set to destination folder ...')
shutil.copytree(args.data, dest_folder)

################################################
### get the len of each class

final_dest_folder = dest_folder + '/train_images'
classes = os.listdir(final_dest_folder)
class_count = {x:len(os.listdir(final_dest_folder + '/' + x)) for x in classes}
flipped_class_clount = dict(class_count)
file_names = {x:os.listdir(final_dest_folder + '/' + x) for x in classes}
print('class count before balancing ', class_count)

################################################
### start by applying flips to balance 
print('flipping')
for c in tqdm(class_count.items()):
    ### if images in the flipping class can be flipped
    if c[0] in flippable and flipped_class_clount[flippable[c[0]]]  < args.min:

        #print('\nflipping', c[0], ' to ', flippable[c[0]])
        origin_class = final_dest_folder + '/' + c[0]
        dest_class = final_dest_folder + '/' + flippable[c[0]]
        files_random = sample(file_names[c[0]], len(file_names[c[0]]))

        flipped = 0
        for k in range(0, max(0, min(class_count[c[0]], args.min - flipped_class_clount[flippable[c[0]]]))):
            img = Image.open(origin_class + '/' + files_random[k]).convert('RGB')
            img_flip = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_flip.save(dest_class + '/F-' + c[0] + '_' + str(k).zfill(5) + '-' + files_random[k])
            flipped += 1

        flipped_class_clount[flippable[c[0]]] += flipped
print('class count after flipping ', flipped_class_clount)

#################################################
### augment classes with geometric transforms

### rebuild list of file names to include flips
file_names = {x:os.listdir(final_dest_folder + '/' + x) for x in classes}
final_class_count = dict(flipped_class_clount)

print('augmenting with geometric transforms')
for c in tqdm(class_count.items()):

    if class_count[c[0]]  < args.min:

        the_class = final_dest_folder + '/' + c[0]

        for k in range(0, args.min - flipped_class_clount[c[0]] + 1):

            ### select a random file
            image = choice(file_names[c[0]])
            img = Image.open(the_class + '/' + image)

            ### rotate up to 10 degrees (image is resized prior to rotation, and scaled back after)
            rot = random() * 20 - 10
            i_size = img.size
            img = img.resize((img.size[0]*2, img.size[1]*2), PIL.Image.BICUBIC)
            img_rot = img.rotate(rot, expand=0)
            img_rot = img_rot.resize(i_size , PIL.Image.LANCZOS)
           
            ### translate up to for pixels
            transx = randint(0,4)
            transy = randint(0,4)
            img_trans = img_rot.rotate(0, translate=[2-transx,2-transy])

            ### save transformed image
            img_trans.save(the_class + '/TR-' + c[0] + '_' + str(k).zfill(5) + '-' + image)
        final_class_count[c[0]] += k

print('final class count :',  final_class_count)

