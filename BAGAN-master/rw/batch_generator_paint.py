"""
(C) Copyright IBM Corporation 2018
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import os, sys
import numpy as np
from random import randint,sample,shuffle
from scipy import ndimage
import csv
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class MnistBatchGenerator:

    TRAIN = 1
    TEST = 0

    def __init__(self, data_src, batch_size=32, class_to_prune=None, unbalance=0):
        self.batch_size = batch_size
        self.data_src = data_src

        # Load data
        
        assert self.batch_size > 0, 'Batch size has to be a positive integer!'

        if self.data_src == self.TEST:
            self.folderimages="../../test_reduced"
            self.csvfile="../../final_reduced_test_info.csv"
        else:
            self.folderimages="../../train_reduced"
            self.csvfile="../../final_reduced_train_info.csv"

        self.dic={"Romanticism": 0, "Realism": 1, "Impressionism": 2}


        self.files=[f for f in os.listdir(self.folderimages) if os.path.isfile(os.path.join(self.folderimages,f))]
        self.files_int=[int(f[:-4]) for f in self.files]
        self.dic2=dict()

        with open(self.csvfile, mode='r') as csv_file:
            csv_reader=csv.DictReader(self.csv_file)
            for row in csv_reader:
                if row["new_filename"] in self.files:
                    self.dic2[row["new_filename"]]=self.dic[row["style"]]

        # Compute per class instance count.
        self.classes=len(self.dic)
        self.per_class_count=list()
        for c in range(self.classes):
            counter=0
            for key, value in self.dic2.items():
                if value==c:
                    counter=counter+1
            self.per_class_count.append(counter)


        # List of labels
        self.label_table = [str(c) for c in range(len(self.dic))]



    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size
        keys=self.dic2.keys()
        shuffle(keys)
        counter=0
        samples_names=list()
        for key in keys:
            if self.dic2[key]==c:
                counter=counter+1
                samples_names.append(key)
            if counter==samples:
                break
    
        xreturn=list()
        for s in range(0,samples):
            img=load_img(self.folderimages+"/"+samples_names[s])
            x=img_to_array(img)
            x=np.rollaxis(x, 2, 0)
            xwidth=x.shape[1]
            xlength=x.shape[2]
            xrandw=randint(0,xwidth-256)
            xrandl=randint(0,xlength-256)
            x=x[:,xrandw:xrandw+256,xrandl:xrandl+256]
            x=(x-128)/128
            xreturn.append(x)
        return np.stack(xreturn,axis=0)
        
    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len(self.label_table)

    def get_class_probability(self):
        return self.per_class_count/sum(self.per_class_count)

    ### ACCESS DATA AND SHAPES ###
    def get_num_samples(self):
        return len(self.files)

    def get_image_shape(self):
        return [3,256,256]

    def next_batch(self):
        s=sample(self.files_int,self.batchsize)
    
        for i in s:
            img=load_img(self.folderimages+"/"+str(i)+'.jpg')
            x=img_to_array(img)
            x=np.rollaxis(x, 2, 0)
            xwidth=x.shape[1]
            xlength=x.shape[2]
            xrandw=randint(0,xwidth-256)
            xrandl=randint(0,xlength-256)
            x=x[:,xrandw:xrandw+256,xrandl:xrandl+256]
            x=(x-128)/128
            yield x, self.dic2[str(i)+".jpg"]


