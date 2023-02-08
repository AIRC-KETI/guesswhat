import h5py
import numpy as np
import os
from PIL import Image
save_path = './numpy.hdf5'
img_path = '1.jpeg'
print('image size: %d bytes'%os.path.getsize(img_path))
hf = h5py.File(save_path, 'a') # open a hdf5 file
img_np = np.array(Image.open(img_path))

dset = hf.create_dataset('default', data=img_np)  # write the data to hdf5 file
hf.close()  # close the hdf5 file
print('hdf5 file size: %d bytes'%os.path.getsize(save_path))