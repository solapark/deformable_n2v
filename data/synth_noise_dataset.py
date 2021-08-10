import torch
import torch.utils.data as data
import os
import cv2
from synthetic_data_generator import AugmentNoise

class synth_noise_dataset(data.Dataset):
    def __init__(self, config, data_name):
        super(synth_noise_dataset, self).__init__()

        dataset_dir = os.path.join(config['data_root'], data_name)
        dir_list = os.listdir(dataset_dir)
        self.path_list = [os.path.join(dataset_dir, filename) for filename in dir_list] 
        self.noise_adder = AugmentNoise(style=config['noise_style'])
        
    def __getitem__(self, index):
        x = cv2.imread(self.path_list[index])
        w, h, _ = x.shape
        w = w - w%16
        h = h - h%16
        x = cv2.resize(x, (h, w))
        x = x.transpose(2, 0, 1)
        self.clean = torch.from_numpy(x).float()/255.
        self.noisy = self.noise_adder.add_valid_noise(self.clean)
        result = {'gt':self.clean, 'noise':self.noisy}
        return result

    def __len__(self):
        return len(self.path_list)
