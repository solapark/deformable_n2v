import h5py
import torch
import torch.utils.data as data
import numpy as np

from synthetic_data_generator import AugmentNoise

class imagenet(data.Dataset):
    def __init__(self, config, _):
        super(imagenet, self).__init__()
        self.h5file = h5py.File(config['imagenet_path'], 'r')
        self.num = self.h5file['images'].shape[0]
        self.crop_size = config['crop_size']

        self.noise_adder = AugmentNoise(style=config['noise_style'])
        
    def __getitem__(self, index):
        x = self.h5file['images'][index]
        shape = self.h5file['shapes'][index]
        x = np.reshape(x, shape)
        x = self.random_crop_numpy(x)
        self.clean = torch.from_numpy(x).float() / 255.
        self.noisy = self.noise_adder.add_valid_noise(self.clean)
        result = {'gt':self.clean, 'noise':self.noisy}
        return result

    def __len__(self):
        return self.num

    def random_crop_numpy(self, img):
        y = np.random.randint(img.shape[1] - self.crop_size + 1)
        x = np.random.randint(img.shape[2] - self.crop_size + 1)
        return img[:, y : y+self.crop_size, x : x+self.crop_size]

