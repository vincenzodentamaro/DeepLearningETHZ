import os
import csv
import shutil
import argparse
import pandas as pd
from skimage import io

# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--outputdir", help='Name of the output directory', type=str, default='train_reduced')
parser.add_argument("--inputdir", help='Name of the input directory', type=str, default='train')

args = parser.parse_args()

outputdir = os.path.join(os.path.curdir,args.outputdir)
inputdir = os.path.join(os.path.curdir,args.inputdir)

# open the file in universal line ending mode
with open('./data_info_files/reduced_train_info.csv', 'rU') as infile:
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

with open('./data_info_files/removed_files.csv', 'w') as removed_file:
    outcsv = csv.writer(removed_file)
    outcsv.writerow(['new_filename'])

for f in files:
    file_input = os.path.join(inputdir,f)
    file_output = os.path.join(outputdir,f)
    if os.path.isfile(file_input):
        if not os.path.isfile(file_output):
            try:
                tt= io.imread(file_input)
                if len(tt.shape)==3:
                    if tt.shape[2] == 3:
                        if min(tt.shape[0], tt.shape[1])>=124:
                            shutil.copy2(file_input, file_output)
                    else:
                        print('{} has {} channels and is thus removed'.format(file_input, tt.shape[2]))
                        with open('./data_info_files/removed_files.csv', 'a') as removed_file:
                            outcsv = csv.writer(removed_file)
                            outcsv.writerow([f])
                else:
                    print('{} is a grayscale image and is thus removed'.format(file_input))
                    with open('./data_info_files/removed_files.csv', 'a') as removed_file:
                        outcsv = csv.writer(removed_file)
                        outcsv.writerow([f])
            except:
                print('{} could not be opened by io.imread'.format(file_input))
                with open('./data_info_files/removed_files.csv','a') as removed_file:
                    outcsv = csv.writer(removed_file)
                    outcsv.writerow([f])


# make new info file with the rows of the skipped files removed from the info file

info_file = pd.read_csv('./data_info_files/reduced_train_info.csv')
removed_files = pd.read_csv('./data_info_files/removed_files.csv')
info_file_removed = info_file.loc[~(info_file.new_filename.isin(removed_files['new_filename'])), :]
info_file_removed.to_csv('./data_info_files/final_train_info.csv', index=False)
