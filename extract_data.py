import os
import pandas as pd
import csv
import shutil


outdir = os.path.abspath(os.path.join(os.path.curdir, "TEST"))
if not os.path.exists(outdir):
            os.makedirs(outdir)

# data.csv is the file where containing artist names, corresponding style and jpg image name
# reduced_data.csv is the file containing only artist names, corresponding style and jpg image name for the styles we are interested in (the ones listed in look_for)


look_for = ['Realism', 'Romanticism', 'Impressionism']

with open('./data.csv','rU') as inf, open('./reduced_data.csv','w') as outf:
    incsv = csv.reader(inf, delimiter=',')
    outcsv = csv.writer(outf, delimiter=',')
    outf.write("artist,style,new_filename\n")
    outcsv.writerows(row for row in incsv if row[1] in look_for)



# open the file in universal line ending mode 
with open('./reduced_data.csv', 'rU') as infile:
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
	file = './train_8/'+f # train_8 is the name of the downloaded data file from Kaggle which we want to extract the relevant paintings from.
	file2 = './TEST/'+f # TEST is the name of the directory where our own usable data will be stored
	if os.path.isfile(file):
		if not os.path.isfile(file2):
			shutil.copy2(file, './TEST')
