#training imagenet
python main.py --config configs/train.yaml --savename imagenet

#calc psnr/ssim of Set14
python main.py --config configs/test.yaml --savename imagenet

#save denoised image of Set14
python main.py --config configs/demo.yaml --savename imagenet
