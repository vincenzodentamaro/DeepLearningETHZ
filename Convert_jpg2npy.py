import os
from skimage import io
import numpy as np

path = './image_folder_test/'
all_images = []
for image_path in os.listdir(path):
    img = io.imread(path + image_path, as_gray=False)
    img = np.array(img)
    all_images.append(img)

np.save('test.npy',all_images[0])



