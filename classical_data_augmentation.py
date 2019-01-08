import numpy as np
from PIL import Image
import os
import argparse
import sys
import shutil
import itertools
import re


np.random.seed(4)


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--outputdir", help='Name of the output directory', type=str, default='train_reduced')
parser.add_argument("--inputdir", help='Name of the input directory', type=str, default='train')

args = parser.parse_args()

outputdir = os.path.join(os.path.curdir,args.outputdir)
inputdir = os.path.join(os.path.curdir,args.inputdir)
#print(inputdir)

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

files = os.listdir(inputdir)
#print(files)


# create image paths list
image_paths = list(map(lambda x: os.path.join(inputdir,x), files))
#print(image_paths)


####### append from command line arguments
######for i in range(1, len(sys.argv)):
######    image_paths.append(sys.argv[i])



def four_crop_and_rescale_images(img):
    w, h = img.size
    left1, top1, right1, bottom1 = 0, 0, w/2, h/2
    left2, top2, right2, bottom2 = w/2, h/2, w, h
    left3, top3, right3, bottom3 = w/2, 0, w, h/2
    left4, top4, right4, bottom4 = 0, h/2, w/2, h
    rescale_width, rescale_height = 224, 224
    first_cropped_image = img.crop((left1, top1, right1, bottom1))
    second_cropped_image = img.crop((left2, top2, right2, bottom2))
    third_cropped_image = img.crop((left3, top3, right3, bottom3))
    fourth_cropped_image = img.crop((left4, top4, right4, bottom4))
    first_rescaled_image = first_cropped_image.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    second_rescaled_image = second_cropped_image.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    third_rescaled_image = third_cropped_image.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    fourth_rescaled_image = fourth_cropped_image.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    return [first_rescaled_image,second_rescaled_image,third_rescaled_image,fourth_rescaled_image]

def flip_and_rescale_image(img):
    rescale_width, rescale_height = 224, 224
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    rescaled_flipped_image = flipped_img.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    return rescaled_flipped_image 

def rescale_image(img):
    rescale_width, rescale_height = 224, 224
    rescaled_image = img.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    return rescaled_image

def random_cropped_image(img):
    height, width = img.size
    crop_width, crop_height = 224, 224
    i = int(np.random.uniform(0,int(width-crop_width)))
    j = int(np.random.uniform(0,int(height-crop_height)))
    left, top, right, bottom = i, j, i+crop_width, j+crop_height
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def apply_transforms(image_path):
    img = Image.open( image_path )
    height, width = img.size
    data = four_crop_and_rescale_images(img)
    data.append(flip_and_rescale_image(img))
    data.append(rescale_image(img))
    if min(height,width) >=224:
        data.append(random_cropped_image(img))
    return(data)


#for idx, i in enumerate(data):
#	i.save('./augmented_dataset/tmp_%d.jpg' % idx)

for f in image_paths:
    image_name = os.path.splitext(os.path.basename(f))[0]
    for idx, i in enumerate(apply_transforms(f)):
        i.save(os.path.join(outputdir,'tmp_%s_%d.jpg') % (image_name, idx))






