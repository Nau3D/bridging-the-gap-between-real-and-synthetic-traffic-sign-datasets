'''
#############################
Scripts used in the making of the paper:

Bridging the Gap Between Real and Synthetic Traffic Sign Repositories
Diogo Lopes da Silva, and António Ramires Fernandes

To appear in proceedings of Delta 2022


#############################
 
Script to generate synthetic datasets

The dataset is augmented such that each class has at least "min" elements. Vertical flips are used primary, taking into account that the flipped sample may belong to another class

Usage example:

python generator.py --templates template_location --output dest_dir --number 2000 --seed 0 --brightness exp2 --negative_folder backgrounds --negative_ratio 1


This commands assumas that in the current folder the following folders have been previously created:

1- template_folder: a folder with the templates for all classes
2- negative-folder: a folder with images to be used as backgrounds (when using real backgrounds)

A folder "dest_dir" will be created with the resulting dataset

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
from random import randint, uniform, random
from time import perf_counter
import random as rd

import noise
import numpy as np
import cv2
import skimage

from PIL import Image, ImageEnhance
from scipy import stats
import multiprocessing as mp
import re
import datetime


### Johnson distribution parameters: these values will be used in the add_background function to determine how brightness is set
johnson = [
    [0.727,1.694,2.893,298.639], # belgium
    [0.664,1.194,20.527,248.357], #croatium
    [0.747,0.907,7.099,259.904]] #german

johnson_index = {'belgium': 0, 'croatian':1, 'german':2}
###############################################################



def load_templates(path):
    num_classes = 0
    for template in os.listdir(f'{path}'):
        match = re.match('[0-9]+(_0)?.png', template)
        if match:
            num_classes += 1

    templates = []
    for i in range(num_classes):
        template = cv2.imread(f'{path}/{i}.png', cv2.IMREAD_UNCHANGED)
        if template is not None:
            templates.append([template])
        else:
            templates.append([cv2.imread(image, cv2.IMREAD_UNCHANGED) for image in sorted(glob.glob(f'{path}/{i}_[0-9]*.png'))])
    return templates


def change_brightness(image, brightness):
    old_brightness = get_image_brightness2(image)
    new_ratio = brightness / old_brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            b = round(v[i][j] * new_ratio)
            if b > 255:
                b = 255
            v[i][j] = b
    # v = v * new_ratio
    output = cv2.cvtColor(cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2BGRA)
    if channels == 4:
        output[:, :, 3] = image[:, :, 3]
    return output


def get_image_brightness2(bgr_image):
    mask = cv2.inRange(bgr_image, (0, 0, 0, 255), (255, 255, 255, 255))
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    hsv_brightness = np.mean(v[mask > 0])
    return hsv_brightness


def paste(foreground, background, position=(0, 0)):
    output = Image.fromarray(background)
    input = Image.fromarray(foreground)
    output.paste(input, position, mask=input)
    return np.asarray(output)


def overlay_transparent(foreground, background, position=(0, 0)):
    x, y = position
    h, w = foreground.shape[:2]

    overlay = foreground[..., :3]
    alpha = foreground[..., 3:] / 255

    background[y: y + h, x: x + w] = (1 - alpha) * background[y: y + h, x: x + w] + alpha * overlay
    return background


def random_erase(image, margin=0.1):
    new_image = image.copy()

    position = randint(0, 2)
    if position == 0:
        hr = (0.1, 0.25)
        wr = (0.1, 0.25)
    elif position == 1:
        hr = (0.1, 0.15)
        wr = (0.15, 0.4)
    else:
        hr = (0.15, 0.4)
        wr = (0.1, 0.15)

    hratio = uniform(hr[0], hr[1])
    wratio = uniform(wr[0], wr[1])
    height, width = image.shape[:2]
    h, w = round(hratio * height), round(wratio * width)

    y = randint(round(height * margin), round((height - h)*(1-margin)))
    x = randint(round(width * margin), round((width - w)*(1-margin)))

    new_image[y: y + h, x: x + w, : 3] = (np.random.rand(h, w, 3) * 255).astype(np.uint8)  #randint(0, 255), randint(0, 255), randint(0, 255)  #

    return new_image


def get_background(height, width, neg_folder):

    index = rd.randint(0, len(bg_list)-1)
    image = cv2.imread(f'{neg_folder}/{bg_list[index]}', cv2.IMREAD_UNCHANGED)

    img_height, img_width, channels = image.shape

    w0 = rd.randint(0, img_width - width - 1)
    h0 = rd.randint(0, img_height - height - 1)

    crop = image[h0: h0+height, w0: w0+width]
    return crop


def add_background(traffic_sign, neg_folder, neg_prob, size=(18, 48), range=(0.07, 0.21), limit=0.1, random_erase_prob=0.0, brightness='exp'):
    s = randint(size[0], size[1])
    h, w = s, s
    image = cv2.resize(traffic_sign, (h, w), interpolation=cv2.INTER_AREA)

    if brightness == 'exp2':
        r = 10 + pow(random(), 2) * 240
    else:
        i = johnson_index[brightness]
        r = stats.johnsonsb.rvs(johnson[i][0], johnson[i][1], johnson[i][2], johnson[i][3])
    ################################################################################################

    image = change_brightness(image, r)
    
    margin = uniform(range[0], range[1])
    height, width = round(h / (-2 * margin + 1)), round(w / (-2 * margin + 1))

    neg_random_prob = random()
    if neg_prob > 0.0 and neg_prob >= neg_random_prob  and neg_folder != None:
        background = get_background(width, height, neg_folder)
    else:   
        background = np.zeros((height, width, 3), np.uint8)
        background[:] = tuple((randint(0, 255), randint(0, 255), randint(0, 255)))


    if (1 - (h / height)) / 2 < limit or (1 - (w / width)) / 2 < limit:
        y = round((height - h) / 2)
        x = round((width - w) / 2)
    else:
        y = randint(round(height * limit), round(height - height * limit - h))
        x = randint(round(width * limit), round(width - width * limit - h))

    output = paste(image, background, (y, x))

    # for images with real backgrounds apply the change in brightness after the composition
    if neg_prob > 0.0 and neg_prob >= neg_random_prob and neg_folder != None:
        output2 = random_erase(output, limit) if random() < random_erase_prob else output
        output3 = change_brightness(output2, r)
        return output3
  
    return random_erase(output, limit) if random() < random_erase_prob else output


def perlin_noise(shape=(1024, 1024), octaves=6):
    scale = 100.0
    persistence = 0.5
    lacunarity = 2.0

    matrix = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix[i][j] = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                         repeatx=1024, repeaty=1024, base=0)
    return cv2.cvtColor(np.round(((matrix - np.min(matrix)) * (1 / (np.max(matrix) - np.min(matrix))) * 255)).astype('uint8'), cv2.COLOR_GRAY2BGR)


def extract_from(image, shape=(512, 512)):
    h, w = image.shape[:2]
    y = randint(0, h - shape[0])
    x = randint(0, w - shape[1])
    return image[y: y + shape[0], x: x + shape[1]]


def perspective_transform(template):
    y, x = template.shape[:2]
    src = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
    quadrant = np.float32([[1,1], [-1, 1], [-1,-1], [1, -1]])

    disp = []
    for i in range(4):
        u = uniform(0.05, 0.25)
        v = uniform(0.05, 0.25)
        disp.append([round(u * x), round(u * y)])
    
    corners = []
    for i in range(4):
        corners.append(src[i] + disp[i]*quadrant[i] )
    corners = np.float32(corners)
    
    M = cv2.getPerspectiveTransform(src, corners)    
    img = cv2.warpPerspective(template, M, (x,y))
    bbox = cv2.boundingRect(corners)  
    img = img[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[2]+bbox[0]]
    return img


def gaussian_noise(image):
    gaussian = skimage.util.random_noise(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    gaussian = cv2.cvtColor(cv2.normalize(gaussian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLOR_BGR2BGRA)
    gaussian[:, :, 3] = image[:, :, 3]
    return gaussian


def motion_blur(image, size=5):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur /= size
    return cv2.filter2D(image, -1, kernel_motion_blur)


def radial_motion_blur(image, size, angle):
    k = np.zeros((size, size))
    k[(size - 1) // 2, :] = np.ones(size)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5 , size / 2 - 0.5 ) , angle, 1), (size, size) )
    k = k * (1 / np.sum(k))
    return cv2.filter2D(image, -1, k)


def confetti_noise(image, kernel=(16, 16), probability=0.03, spacing=8):
    height, weight = image.shape[:2]
    kh, kw = kernel
    y, x = 0, 0
    for i in range(height - kh):
        if y > 0:
            y -= 1
            continue
        for j in range(weight - kw):
            if x > 0:
                x -= 1
                continue
            if random() <= probability:
                image[i: i + kh, j: j + kw, : 3] = randint(0, 255), randint(0, 255), randint(0, 255)
                y, x = kh + spacing, kw + spacing
    return image


def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * factor
    result = cv2.cvtColor(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = image[:, :, 3]
    return result


def add_perlin_noise(traffic_sign, perlin_noise, alpha=0.5):
    y, x = traffic_sign.shape[:2]
    mask = cv2.inRange(traffic_sign, (0, 0, 0, 255), (255, 255, 255, 255))
    perlin = cv2.resize(extract_from(perlin_noise), (x, y), interpolation=cv2.INTER_AREA)
    output = np.zeros((y, x, 3), np.uint8)
    cv2.addWeighted(cv2.cvtColor(traffic_sign, cv2.COLOR_BGRA2BGR), alpha, perlin, 1 - alpha, 0, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
    traffic_sign[mask > 0] = output[mask > 0]
    return traffic_sign


def rotate(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((h / 2, w / 2), randint(0, angle * 2) - angle, 1)
    return cv2.warpAffine(image, M, (h, w))


def shadow_from_perlin(perlin):
    h, w = perlin.shape[:2]
    perlin_shadow = cv2.cvtColor(perlin, cv2.COLOR_BGR2BGRA)
    for i in range(h):
        for j in range(w):
            if perlin_shadow[i][j][0] > 119:
                perlin_shadow[i][j] = [255, 255, 255, 0]
            else:
                perlin_shadow[i][j] = [0, 0, 0, 255]
    return perlin_shadow


def insert_shadow(image, perlin_shadow):
    h, w = image.shape[:2]
    shadow = cv2.cvtColor(extract_from(perlin_shadow, (512, 512)), cv2.COLOR_BGR2BGRA)
    shadow[:,:, 3] = shadow[:,:, 3] * uniform(0.60, 0.75)

    shadow = cv2.resize(shadow, (h, w), interpolation=cv2.INTER_AREA)
    return paste(shadow, image)


def shear_transform(image, shear=(0.03, 0.10)):
    low, high = shear
    h, w = image.shape[:2]

    src = np.float32([[0, 0], [w, 0], [w, h]])
    operation = randint(0, 3)
    if operation == 0:
        dst = np.float32([[w * uniform(low, high), 0], [w + w *  uniform(low, high), 0], [w - w * uniform(low, high), h ]])
    elif operation == 1:
        dst = np.float32([[- w * uniform(low, high), 0], [w - w * uniform(low, high), 0], [w + w * uniform(low, high), h]])
    elif operation == 2:
        dst = np.float32([[0, h * uniform(low, high)], [w, - h * uniform(low, high)], [w, h - h * uniform(low, high)]])
    else:
        dst = np.float32([[0, - h * uniform(low, high)], [w, + h * uniform(low, high)], [w, h + h * uniform(low, high)]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, M, (w, h))

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3) , 0)


def hsv_jitter(image, hue=(-12, 20), saturation=(0.4, 2), brightness=(0.5, 1), probs=(0.8, 0.8, 0)):
    if random() < probs[0]:
        shift = uniform(0, hue[1] / 2 - hue[0] / 2) + hue[0] / 2
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0] + shift) % 180
        hue_shifted = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2BGRA)
        hue_shifted[:, :, 3] = image[:, :, 3]
        image = hue_shifted
    if random() < probs[1]:
        enhanced = ImageEnhance.Color(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))).enhance(uniform(saturation[0], saturation[1]))
        image = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_BGRA2RGBA)
    if random() < probs[2]:
        enhanced = ImageEnhance.Brightness(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))).enhance(uniform(brightness[0], brightness[1]))
        image = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_BGRA2RGBA)
    return image


def gen(proc, classes, templates, num_per_class, output_path, perlin_noise, perlin_alpha, shear_prob, perlin_shadow, seed, brightness_param, confetti_prob, neg_folder, neg_prob):
    rd.seed(seed)
    np.random.seed(seed)
    start = perf_counter()
    print(f'Process {proc}: {list(classes)[0]}..{list(classes)[-1]}')

    global bg_list
    bg_list = []
    if neg_prob > 0 and neg_folder != None:
        bg_list = os.listdir(neg_folder)


    for class_id in classes:
        for i in range(round(num_per_class * 0.8)):
            #alpha = 0.6

            num_templates = len(templates[class_id])
            image = templates[class_id][randint(0, num_templates-1)].copy()
            image = hsv_jitter(image)

            if random() < 0.7:
                image = rotate(image, 15)
            if random() < 0.6:
                image = perspective_transform(image)
            if random() < shear_prob:
                image = shear_transform(image)

            if not perlin_alpha == 1:
                image = add_perlin_noise(image, perlin_noise, perlin_alpha)

            image = add_background(image, neg_folder, neg_prob, (18, 48), (0.07, 0.21), 0.14, 0.0, brightness_param)  #, brightness=not alpha == 1

            if random() < 0.3:
                image = motion_blur(image, randint(2, 5))

            cv2.imwrite(f'{output_path}/{format(class_id, "05d")}/{i}.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        for i in range(round(num_per_class * 0.2)):
            num_templates = len(templates[class_id])
            image = templates[class_id][randint(0, num_templates-1)].copy()

            image = hsv_jitter(image)

            if random() < 0.7:
                image = rotate(image, 15)
            if random() < 0.6:
                image = perspective_transform(image)
            if random() < shear_prob:
                image = shear_transform(image)

            image = add_perlin_noise(image, perlin_noise, perlin_alpha)

            if random() < confetti_prob:
                image = confetti_noise(image)

            image = add_background(image, neg_folder, neg_prob, (12, 17), (0.20, 0.25), 0.19, 0.0, brightness_param)

            cv2.imwrite(f'{output_path}/{format(class_id, "05d")}/{i + round(num_per_class * 0.8)}.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f'Thread {proc}: {class_id}')
    print(f'Thread {proc}: {str(datetime.timedelta(seconds=perf_counter() - start)).split(".")[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic traffic sign generator')
    parser.add_argument('-t', '--templates', type=str, help="template input directory", metavar='/path', required=True)
    parser.add_argument('-o', '--output', type=str, help='where to generate traffic signs directory', metavar='/path', required=True)
    parser.add_argument('-n', '--number', type=int, default=2000, metavar='N', help='number of generated traffic signs per class (default: 2000)')
    parser.add_argument('-s','--seed', type=int, default=0, metavar='N', help='random seed for deterministic results (default: %(default)s)')
    parser.add_argument('-p','--processes', type=int, default=3, metavar='N', help='number of processes (default: %(default)s)')
    parser.add_argument('--brightness', choices=['exp2', 'belgium', 'croatian', 'german'], required=True, 
                    help="brightness adjustment. ")
    parser.add_argument('--perlin_alpha', type=float, default=0.6, help='1 - intensity of perlin noise (default: %(default)s)')
    parser.add_argument('--shear_prob', type=float, default=0.0, help='probability of applying shear (default: %(default)s)')
    parser.add_argument('--confetti_prob', type=float, default=0.5, help='probability of applying confetti to smaller samples (default: %(default)s)')
    parser.add_argument('--negative_folder', type=str, default=None, help='folder with real negative images for backgrounds')
    parser.add_argument('--negative_ratio', type=float, default=0.0, help='probability of using real backgrounds')

    args = parser.parse_args()

    templates_path = args.templates
    output_path = args.output
    num_per_class = args.number
    num_proc = args.processes
    seed = args.seed

    if args.negative_folder == None:
        neg_folder = ''
    else:
        neg_folder = args.negative_folder

    ### Output path is the folder where the dataset folder will be stored
    # this allows to quickly identify the params that generated a given dataset
    brightness_param = args.brightness    
    output_path = (args.output + '/train_gen3.5_' + templates_path.replace('/','_').replace('\\', '_') + '_n' + str(num_per_class) +
             '_s' + str(seed)  + '_p' + str(num_proc)  + '_nf' + neg_folder + '_rp' + str(args.negative_ratio) + '_b' + brightness_param + '_pi' + str(args.perlin_alpha) + '_sp' + str(args.shear_prob) + '_cb' + str(args.confetti_prob) + '/train_images')



    if args.processes != parser.get_default('processes') or args.seed != parser.get_default('seed'):
        print('To ensure deterministic results, the number of processes and seed should be the same each run.')

    if os.path.exists(output_path):
        exit('Data set directory already exists.')

    if not os.path.exists(templates_path):
        exit('Templates directory does not exist.')

    templates = load_templates(templates_path)
    num_classes = len(templates)
    print('\nTemplates loaded:', num_classes, 'classes')

    os.makedirs(output_path)
    [os.makedirs(os.path.join(output_path, format(class_id, "05d"))) for class_id in range(num_classes)]
    print(f'Output directory created: {output_path}')

    perlin_noise = perlin_noise((2048, 2048))
    print('Perlin noise generated')

    processes = [mp.Process(
                    target=gen, 
                    args=(i, 
                        range(round(i * (num_classes / num_proc)), round((i + 1) * (num_classes / num_proc))), 
                        templates, 
                        num_per_class, 
                        output_path, 
                        perlin_noise, 
                        args.perlin_alpha, 
                        args.shear_prob, 
                        None, 
                        seed, 
                        brightness_param, 
                        args.confetti_prob,
                        args.negative_folder,
                        args.negative_ratio
                    )
                ) for i in range(num_proc)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("That's all folks!")

