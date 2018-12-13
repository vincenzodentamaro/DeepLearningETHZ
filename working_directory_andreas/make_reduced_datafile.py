import os
import csv
import argparse


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--outputfile1", help='Name of the outputfile', type=str, default='reduced_train_info.csv')
parser.add_argument("--outputfile2", help='Name of the outputfile', type=str, default='reduced_test_info.csv')
parser.add_argument("--inputfile", help='Name of the inputfile', type=str, default='all_data_info.csv')
args = parser.parse_args()
if args.outputfile1[-4:] != '.csv':
    raise ValueError('Expecting a csv file as output filename, got {} as extension'.format(args.outputfile[-4:]))
if args.inputfile[-4:] != '.csv':
    raise ValueError('Expecting a csv file as input filename, got {} as extension'.format(args.inputfile[-4:]))

outputfile1 = args.outputfile1
outputfile2 = args.outputfile2
inputfile = args.inputfile

# Create reduced datafile
look_for = ['Realism', 'Romanticism', 'Impressionism']

with open('./'+inputfile,'r',encoding='utf-8') as inf, open('./'+outputfile1,'w', newline='') as outf:
    incsv = csv.reader(inf, delimiter=',')
    outcsv = csv.writer(outf, delimiter=',')
    outf.write("artist,style,new_filename\n")
    outcsv.writerows([row[0], row[7], row[-1]] for row in incsv if (row[7] in look_for and row[-2]=='True'))

with open('./'+inputfile,'r',encoding='utf-8') as inf, open('./'+outputfile2,'w', newline='') as outf:
    incsv = csv.reader(inf, delimiter=',')
    outcsv = csv.writer(outf, delimiter=',')
    outf.write("artist,style,new_filename\n")
    outcsv.writerows([row[0], row[7], row[-1]] for row in incsv if (row[7] in look_for and row[-2]=='False'))

