import numpy as np 
import os
import math
from glob import glob
import scipy.misc

crop_size = 224
number_test_images = 100

for h in range(number_test_images):
  x = scipy.misc.imread('./samples/test_arange_%s.png' % (h)).astype(np.float)
  for idx1 in range(8):
    for idx2 in range(8):
      i = idx1*crop_size
      j = idx2*crop_size    				 
      image = x[i:i+crop_size, j:j+crop_size]
      scipy.misc.imsave('./augmented_dataset/test_arange_%s_%s.png' % (idx1, idx2), image)
