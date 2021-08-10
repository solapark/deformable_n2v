import torch
import torch.utils.data as data
import os
import cv2

class real_noise_dataset(data.Dataset):
    def __init__(self, config, _):
        super(real_noise_dataset, self).__init__()

        dataset_dir = os.path.join(config['data_root'], config['data_name'])
        dir_list = os.listdir(dataset_dir)
        self.path_list = [os.path.join(dataset_dir, filename) for filename in dir_list] 
        
    def __getitem__(self, index):
        x = cv2.imread(self.path_list[index])
        x = x.transpose(2, 0, 1)
        self.clean = torch.from_numpy(x).float()
        self.noisy = None
        result = {'gt':self.clean, 'noise':self.noisy}
        return result

    def __len__(self):
        return len(self.path_list)
