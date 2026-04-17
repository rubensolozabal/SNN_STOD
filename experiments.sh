# Train
CUDA_VISIBLE_DEVICES=4 python train.py -model 'vgg11' -dataset 'cifar10' -p 8 -gor_lambda 0.05 #90.43
CUDA_VISIBLE_DEVICES=3 python train.py -model 'vgg11' -dataset 'cifar100' -p 8 -gor_lambda 0.05
CUDA_VISIBLE_DEVICES=4 python train.py -model 'spiking_nfresnet18' -dataset 'imagenet' -T 4 -lr 0.1 -lr_orth 0.05 -weight_decay 1e-5 -epochs 150 -b 512 -p 16 -gor_lambda 0.05


# Adversarial training 
CUDA_VISIBLE_DEVICES=1 python train.py -model vgg11 -dataset cifar10 -attack fgsm -eps 8 -p 8 -gor_lambda 0.05
CUDA_VISIBLE_DEVICES=2 python train.py -model vgg11 -dataset cifar10 -attack pgd -eps 8 -steps 7 -p 8 -gor_lambda 0.05


# Attack inference
CUDA_VISIBLE_DEVICES=0 python test.py -resume logs/cifar10_vgg11_T4_or0.5_tau1.1_e200_bs128_wd0.0005_reg0.05_clean_p8/checkpoint_max.pth -model vgg11 -dataset cifar10 -attack fgsm -eps 8 -p 8
CUDA_VISIBLE_DEVICES=0 python test.py -resume logs/cifar100_vgg11_T4_or0.5_tau1.1_e200_bs128_wd0.0005_reg0.05_clean_p8/checkpoint_max.pth -model 'vgg11' -dataset 'cifar100' -attack fgsm -eps 8 -p 8
CUDA_VISIBLE_DEVICES=0 python test.py -resume logs/cifar10_vgg11_T4_or0.5_tau1.1_e200_bs128_wd0.0005_reg0.05_clean_p8/checkpoint_max.pth -model vgg11 -dataset cifar10 -attack pgd -eps 8 -steps 7 -p 8
CUDA_VISIBLE_DEVICES=0 python test.py -resume logs/cifar100_vgg11_T4_or0.5_tau1.1_e200_bs128_wd0.0005_reg0.05_clean_p8/checkpoint_max.pth -model 'vgg11' -dataset 'cifar100' -attack pgd -eps 8 -steps 7 -p 8

