import argparse
import random
import numpy as np
from utils import numpy_compat
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
from utils.augmentation import ToPILImage, Resize, ToTensor
from torch.utils.data import SubsetRandomSampler
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import surrogate as surrogate_self
from modules import neuron
from models import spiking_resnet_imagenet, spiking_resnet, spiking_vgg_bn


class HuggingFaceImageDataset(data.Dataset):
    def __init__(self, dataset, transform, image_column='image', label_column='label'):
        self.dataset = dataset
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample[self.image_column]
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            image = self.to_pil(image).convert('RGB')
        label = int(sample[self.label_column])
        return self.transform(image), label


def get_args():
    parser = argparse.ArgumentParser(description='Structural Temporal Orthogonal Decorrelation for Robust SNN')
    parser.add_argument('-seed', default=2025, help='Hope you have a lucky 2025 :)', type=int)
    parser.add_argument('-name', default='', type=str, help='specify a name for the checkpoint and log files')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str, help='should be cifar10, cifar100, or imagenet')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-surrogate', default='tri', type=str, help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-pre_train', type=str, help='load a pretrained model. used for imagenet')
    parser.add_argument('-amp', action='store_false', help='automatic mixed precision training')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-lr_orth', default=0.5, type=float, help='learning rate for orthagonalization (Q) parameters')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=200, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='vgg11', help='use which SNN model')
    parser.add_argument('-drop_rate', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-save_init', action='store_true', help='save the initialization of parameters')
    parser.add_argument('-loss_lambda', type=float, default=0.0,  help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-encoding', default='stod', type=str, help='input encoding to use: stod or hypergeometric')

    # STOD
    parser.add_argument('-gor_lambda', type=float, default=0.05,)
    parser.add_argument('-p', type=int, default=8, help='number of patch')

    # Adv
    parser.add_argument('-attack', default='', type=str, help='attack mode: empty, fgsm, or pgd')
    parser.add_argument('-eps', default=2, type=float, metavar='N', help='attack eps')

    # PGD
    parser.add_argument('-alpha', default=0.01, type=float, metavar='N', help='pgd attack alpha')
    parser.add_argument('-steps', default=7, type=int, metavar='N', help='pgd attack steps')
    parser.add_argument(
        '-imagenet_backend',
        default=os.environ.get('IMAGENET_BACKEND', 'auto'),
        choices=('auto', 'imagefolder', 'huggingface'),
        help='ImageNet loader backend: local ImageFolder, Hugging Face dataset, or auto-detect',
    )
    parser.add_argument(
        '-imagenet_hf_path',
        default=os.environ.get('IMAGENET_HF_PATH', ''),
        type=str,
        help='optional local path for a Hugging Face ImageNet dataset saved with load_from_disk',
    )
    parser.add_argument(
        '-imagenet_hf_name',
        default=os.environ.get('IMAGENET_HF_NAME', 'imagenet-1k'),
        type=str,
        help='Hugging Face dataset name to use when imagenet_backend resolves to huggingface',
    )
    parser.add_argument(
        '-imagenet_hf_train_split',
        default=os.environ.get('IMAGENET_HF_TRAIN_SPLIT', 'train'),
        type=str,
        help='train split name for the Hugging Face ImageNet dataset',
    )
    parser.add_argument(
        '-imagenet_hf_val_split',
        default=os.environ.get('IMAGENET_HF_VAL_SPLIT', 'validation'),
        type=str,
        help='validation split name for the Hugging Face ImageNet dataset',
    )
    parser.add_argument(
        '-imagenet_hf_cache_dir',
        default=os.environ.get('IMAGENET_HF_CACHE_DIR', os.environ.get('HF_HOME', '')),
        type=str,
        help='cache directory used by Hugging Face datasets when downloading/loading ImageNet',
    )

    args = parser.parse_args()
    print(args)

    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)

    return args

def _load_hf_split(hf_datasets, dataset_path, split):
    candidate_paths = []
    if dataset_path:
        dataset_path = os.path.expanduser(dataset_path)
        candidate_paths.append(dataset_path)
        candidate_paths.append(os.path.join(dataset_path, split))

    load_errors = []
    for candidate in candidate_paths:
        if not os.path.exists(candidate):
            continue
        try:
            loaded = hf_datasets.load_from_disk(candidate)
        except Exception as exc:
            load_errors.append(f'{candidate}: {exc}')
            continue
        if isinstance(loaded, hf_datasets.DatasetDict):
            if split not in loaded:
                available_splits = ', '.join(loaded.keys())
                raise KeyError(f"Split '{split}' not found in '{candidate}'. Available splits: {available_splits}")
            return loaded[split]
        return loaded

    if load_errors:
        joined_errors = '; '.join(load_errors)
        raise RuntimeError(f'Failed to load Hugging Face dataset from disk: {joined_errors}')

    raise FileNotFoundError(
        f'Could not find a Hugging Face dataset at {dataset_path!r}. '
        f'Expected a dataset root or split directory containing {split!r}.'
    )


def _get_imagenet_hf_dataset(
    dataset_root,
    normalize,
    b,
    j,
    imagenet_hf_path='',
    imagenet_hf_name='imagenet-1k',
    imagenet_hf_train_split='train',
    imagenet_hf_val_split='validation',
    imagenet_hf_cache_dir='',
):
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(
            'Hugging Face ImageNet loading requires the `datasets` package. '
            'Install it with `pip install datasets pyarrow`.'
        ) from exc

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    cache_dir = os.path.expanduser(imagenet_hf_cache_dir) if imagenet_hf_cache_dir else None
    dataset_candidates = []
    if imagenet_hf_path:
        dataset_candidates.append(imagenet_hf_path)
    dataset_candidates.append(dataset_root)
    dataset_candidates.append(os.path.dirname(dataset_root))

    train_dataset = None
    val_dataset = None
    last_error = None
    for candidate in dataset_candidates:
        if not candidate:
            continue
        try:
            train_dataset = _load_hf_split(hf_datasets, candidate, imagenet_hf_train_split)
            val_dataset = _load_hf_split(hf_datasets, candidate, imagenet_hf_val_split)
            print(f'Using Hugging Face ImageNet dataset from disk: {os.path.expanduser(candidate)}')
            break
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            last_error = exc

    if train_dataset is None or val_dataset is None:
        if imagenet_hf_path:
            raise RuntimeError(
                f'Could not load Hugging Face ImageNet dataset from {imagenet_hf_path!r}: {last_error}'
            )
        print(
            f'Local Hugging Face ImageNet dataset not found under {dataset_root}. '
            f'Falling back to load_dataset({imagenet_hf_name!r}).'
        )
        train_dataset = hf_datasets.load_dataset(
            imagenet_hf_name,
            split=imagenet_hf_train_split,
            cache_dir=cache_dir,
        )
        val_dataset = hf_datasets.load_dataset(
            imagenet_hf_name,
            split=imagenet_hf_val_split,
            cache_dir=cache_dir,
        )

    train_data_loader = torch.utils.data.DataLoader(
        HuggingFaceImageDataset(train_dataset, transform_train),
        batch_size=b,
        shuffle=True,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        HuggingFaceImageDataset(val_dataset, transform_val),
        batch_size=b,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )
    return train_data_loader, test_data_loader


def get_data(
    b,
    j,
    T,
    data_dir,
    dataset='cifar10',
    imagenet_backend='auto',
    imagenet_hf_path='',
    imagenet_hf_name='imagenet-1k',
    imagenet_hf_train_split='train',
    imagenet_hf_val_split='validation',
    imagenet_hf_cache_dir='',
):
    if dataset == 'cifar10' or dataset == 'cifar100':
        c_in = 3
        if dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
        train_data_loader = data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=j, drop_last=True)

        testset = dataloader(root=data_dir, train=False, download=False, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=j, drop_last=True)



    elif dataset == 'imagenet':
        num_classes = 1000
        c_in = 3
        normalization_mean = (0.485, 0.456, 0.406)
        normalization_std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=normalization_mean,
                                         std=normalization_std)
        dataset_root = os.path.join(data_dir, 'imagenet')
        traindir = os.path.join(dataset_root, 'train')
        valdir = os.path.join(dataset_root, 'val')

        use_imagefolder = imagenet_backend == 'imagefolder'
        if imagenet_backend == 'auto' and os.path.isdir(traindir) and os.path.isdir(valdir):
            use_imagefolder = True

        if use_imagefolder:
            train_data_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=b, shuffle=True,
                num_workers=j, pin_memory=True, drop_last=True)

            test_data_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=b, shuffle=False,
                num_workers=j, pin_memory=True, drop_last=True)
        else:
            if imagenet_backend == 'imagefolder':
                raise FileNotFoundError(
                    f'Expected ImageNet folders at {traindir} and {valdir}, but they were not found.'
                )
            train_data_loader, test_data_loader = _get_imagenet_hf_dataset(
                dataset_root=dataset_root,
                normalize=normalize,
                b=b,
                j=j,
                imagenet_hf_path=imagenet_hf_path,
                imagenet_hf_name=imagenet_hf_name,
                imagenet_hf_train_split=imagenet_hf_train_split,
                imagenet_hf_val_split=imagenet_hf_val_split,
                imagenet_hf_cache_dir=imagenet_hf_cache_dir,
            )

    else:
        raise NotImplementedError

    return train_data_loader, test_data_loader, c_in, num_classes, normalization_mean, normalization_std

def get_net(surrogate='tri', dataset='cifar10', model='vgg5', num_classes=10, drop_rate=0.0, tau=1.5, c_in=3):
    if surrogate == 'sig':
        surrogate_function = surrogate_sj.Sigmoid()
    elif surrogate == 'rec':
        surrogate_function = surrogate_self.Rectangle()
    elif surrogate == 'tri':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()

    neuron_model = neuron.BPTTNeuron

    if dataset == 'cifar10' or dataset == 'cifar100':
        if 'res' in model:
            if model not in spiking_resnet.__dict__:
                raise KeyError(f"Unknown CIFAR ResNet model '{model}'. Available models include: {[k for k in spiking_resnet.__dict__.keys() if k.startswith('spiking_')]}")
            net = spiking_resnet.__dict__[model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=drop_rate, tau=tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
            print('using Resnet model.')
        elif 'vgg' in model:
            if model not in spiking_vgg_bn.__dict__:
                raise KeyError(f"Unknown CIFAR VGG model '{model}'. Available models include: {[k for k in spiking_vgg_bn.__dict__.keys() if k.startswith('spiking_') or k.startswith('vgg')]}")
            net = spiking_vgg_bn.__dict__[model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=drop_rate, tau=tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
            print('using VGG model.')

    elif dataset == 'imagenet':
        if model not in spiking_resnet_imagenet.__dict__:
            raise KeyError(
                f"Unknown ImageNet model '{model}'. Available models include: "
                f"{[k for k in spiking_resnet_imagenet.__dict__.keys() if k.startswith('spiking_')]}"
            )
        net = spiking_resnet_imagenet.__dict__[model](
            neuron=neuron_model,
            num_classes=num_classes,
            neuron_dropout=drop_rate,
            tau=tau,
            surrogate_function=surrogate_function,
            c_in=3,
        )
        print('using NF-Resnet model.')


    else:
        raise NotImplementedError
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()
    return net
