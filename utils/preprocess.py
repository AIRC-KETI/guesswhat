import h5py
import torch

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(dataset_h5, self).__init__()

        self.file = h5py.File(in_file, 'r')
        self.n_images, self.nx, self.ny = self.file['images'].shape

    def __getitem__(self, index):
        input = self.file['images'][index,:,:]
        return input.astype('float32')

    def __len__(self):
        return self.n_images