import csv
import os
import pandas as pd
import argparse
import numpy as np 

# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--inputdir", help='Name of the input directory', type=str, default='train')
parser.add_argument("--outcsv_file", help= 'name of the csv file with the sample output information', default='outcsv_file.csv')
parser.add_argument("--style", help='Name of the style', type=str)
args = parser.parse_args()
inputdir = os.path.join(os.path.curdir,args.inputdir)
outcsv_file = os.path.join(os.path.curdir,args.outcsv_file)

files = os.listdir(inputdir)
#print(files)

#print(csv_data)
csv_data = pd.DataFrame([['xxx', 'yyy', '123.jpg']], columns=['artist','style','new_filename'])
columnsTitles = ['artist','style','new_filename']

d = []

for f in files:
  #print(f)
  d.append({'artist': 'xxx', 'style': args.style, 'new_filename': f})
  csv_data = pd.DataFrame(d)
  #add = pd.DataFrame([['xxx', args.style, f]], columns=['artist','style','new_filename'])
  #csv_data.append({'artist': 'xxx', 'style': args.style, 'new_filename': f}, ignore_index=True)
  #print(add)
  #print(csv_data)
csv_data.drop(csv_data[csv_data['style'] == 'yyy'].index, inplace=True)
csv_data = csv_data.reindex(columns=columnsTitles)
csv_data.to_csv(outcsv_file, index=False)