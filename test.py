import os
os.system(' python ../test.py --dataroot /smc/jianyao/datasets/image_matrix --name /smc/jianyao/checkpoints/differ_cls --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch --input_nc 7 --output_nc 1')

