import torch
import numpy as np

operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator() 
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator
class AugmentNoise(object):
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [float(p) / 255.0 for p in style.replace('gauss', '', 1).split('_')]
            #self.params = [float(p) for p in style.replace('gauss', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):  
            self.params = [float(p) for p in style.replace('poisson', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"
    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            shape = (shape[0], 1, 1, 1)
            std = self.params[0]
            std = std * torch.ones(shape, device=x.device)
            noise =  np.random.normal(size=shape) * std
            noise = torch.from_numpy(noise).type(torch.FloatTensor) if x.device.type == 'cpu' else torch.from_numpy(noise).type(torch.cuda.FloatTensor)
            result = x + noise
            result = torch.clamp(result, 0, 1)
            #result = torch.clamp(result, 0, 255)
            return result
            #noise = torch.cuda.FloatTensor(shape, device=x.device)
            '''
            noise = torch.FloatTensor(shape, device=x.device) if x.device.type=='cpu' else torch.cuda.FloatTensor(shape, device=x.device)
            #torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            torch.normal(mean=0.0, std=std, generator=None, out=noise)
            return x + noise
            '''
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device) if x.device.type=='cpu' else torch.cuda.FloatTensor(shape, device=x.device)
            #torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            torch.normal(mean=0, std=std, generator=None, out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            #noised = torch.poisson(lam * x, generator=get_generator()) / lam
            noised = torch.poisson(lam * x, generator=None) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
            #noised = torch.poisson(lam * x, generator=get_generator()) / lam
            noised = torch.poisson(lam * x, generator=None) / lam
            return noised
    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            #return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
            noise =  np.random.normal(size=shape) * std
            noise = torch.from_numpy(noise).type(torch.FloatTensor) if x.device.type == 'cpu' else torch.from_numpy(noise).type(torch.cuda.FloatTensor)
            result =x + noise 
            #result = torch.clamp(result, 0, 255)
            result = torch.clamp(result, 0, 1)
            return result
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)

