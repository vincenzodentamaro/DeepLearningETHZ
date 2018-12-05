import os
import csv
import shutil
import argparse
import pandas as pd
import numpy as np
import scipy.misc
from skimage import io

# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--outputdir", help='Name of the output directory', type=str, default='train_reduced')
parser.add_argument("--inputdir", help='Name of the input directory', type=str, default='train')

args = parser.parse_args()

outputdir = os.path.join(os.path.curdir,args.outputdir)
inputdir = os.path.join(os.path.curdir,args.inputdir)

# open the file in universal line ending mode
with open('./reduced_train_info.csv', 'rU') as infile:
    # read the file as a dictionary for each row ({header : value})
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

# extract the variables you want
files = data['new_filename']

for f in files:
    file_input = os.path.join(inputdir,f)
    file_output = os.path.join(outputdir,f)
    if os.path.isfile(file_input):
        if not os.path.isfile(file_output):
            try:
                tt= io.imread(file_input)
                if len(tt.shape)==3:
                    shutil.copy2(file_input, file_output)
            except:
                print('{} could not be opened by io.imread'.format(file_input))
                with open('./removed_files.csv','a') as removed_file:
                    outcsv = csv.writer(removed_file)
                    outcsv.writerow(f)


        # else:
        #     raise RuntimeWarning('file present already present in outputfolder: {}'.format(file_input))

# # artists is a list containing the artist labels in the right order and artist is the same but with a numpy array structure
# t = pd.read_csv('reduced_data.csv')
# artists = t['artist'].tolist()
# artist = np.asarray(artists)
# print(artist)

#
# def is_grey_scale(img_path="lena.jpg"):
#     im = Image.open(img_path).convert('RGB')
#     w,h = im.size
#     for i in range(w):
#         for j in range(h):
#             r,g,b = im.getpixel((i,j))
#             if r != g != b: return False
#     return True