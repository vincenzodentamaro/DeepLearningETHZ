import os
import csv
import argparse


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--outputfile", help='Name of the outputfile', type=str, default='./data_info_files/reduced_train_info.csv')
parser.add_argument("--inputfile", help='Name of the inputfile', type=str, default='./data_info_files/train_info.csv')
args = parser.parse_args()
if args.outputfile[-4:] != '.csv':
    raise ValueError('Expecting a csv file as output filename, got {} as extension'.format(args.outputfile[-4:]))
if args.inputfile[-4:] != '.csv':
    raise ValueError('Expecting a csv file as input filename, got {} as extension'.format(args.inputfile[-4:]))

outputfile = args.outputfile
inputfile = args.inputfile

# Create reduced datafile
look_for = ['Realism', 'Romanticism', 'Impressionism']

with open('./'+inputfile,'r',encoding='utf-8') as inf, open('./'+outputfile,'w', newline='') as outf:
    incsv = csv.reader(inf, delimiter=',')
    outcsv = csv.writer(outf, delimiter=',')
    outf.write("artist,style,new_filename\n")
    outcsv.writerows([row[1], row[-3], row[0]] for row in incsv if row[-3] in look_for)



