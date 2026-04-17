import os

import torch

from utils import numpy_compat
from utils.config import get_args, get_data, get_net
from utils.tvc import PatchwiseQModule, val


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_attacker(args, model):
    if not args.attack:
        return None

    try:
        import torchattacks
    except ImportError as exc:
        raise ImportError("torchattacks is required for adversarial evaluation. Install it before using -attack fgsm or -attack pgd.") from exc

    if args.attack == 'fgsm':
        return torchattacks.FGSM(model, eps=args.eps / 255)
    if args.attack == 'pgd':
        return torchattacks.PGD(model, eps=args.eps / 255, alpha=args.alpha, steps=args.steps)
    raise ValueError(f"Unsupported attack: {args.attack}")


def attach_patch_module(model, data_loader, p, time_step):
    if hasattr(model, '_patch_module'):
        return

    frame0, _ = next(iter(data_loader))
    _, c_in, _, _ = frame0.shape
    patch_module = PatchwiseQModule(c_in, p, time_step).cuda()
    model._patch_module = patch_module
    model.Q = patch_module.Qs


def load_model_weights(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('net') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)
    return checkpoint, incompatible


def describe_attack(args):
    if args.attack == 'fgsm':
        return f'fgsm (eps={args.eps})'
    if args.attack == 'pgd':
        return f'pgd (eps={args.eps}, alpha={args.alpha}, steps={args.steps})'
    return 'clean'


def main():
    args = get_args()
    if not args.resume:
        raise ValueError('Please provide a checkpoint path with -resume for evaluation.')

    _, test_data_loader, c_in, num_classes = get_data(args.b, args.j, args.T, args.data_dir, args.dataset)
    net = get_net(args.surrogate, args.dataset, args.model, num_classes, args.drop_rate, args.tau, c_in)
    attach_patch_module(net, test_data_loader, args.p, args.T)

    checkpoint, incompatible = load_model_weights(net, args.resume)

    if incompatible.missing_keys:
        print(f'Missing keys while loading checkpoint: {incompatible.missing_keys}')
    if incompatible.unexpected_keys:
        print(f'Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}')

    attacker = build_attacker(args, net)
    test_loss, test_acc = val(
        model=net,
        dataset=args.dataset,
        data=test_data_loader,
        time_step=args.T,
        epoch=0,
        optimizer=None,
        lr_scheduler=None,
        scaler=None,
        loss_lambda=args.loss_lambda,
        attacker=attacker,
        writer=None,
    )

    epoch = checkpoint.get('epoch', 'n/a') if isinstance(checkpoint, dict) else 'n/a'
    max_val_acc = checkpoint.get('max_val_acc', 'n/a') if isinstance(checkpoint, dict) else 'n/a'

    print(f'checkpoint={args.resume}')
    print(f'evaluation_attack={describe_attack(args)}')
    print(f'checkpoint_epoch={epoch}')
    print(f'checkpoint_max_val_acc={max_val_acc}')
    print(f'test_loss={test_loss:.4f}')
    print(f'test_acc={test_acc:.4f}')


if __name__ == '__main__':
    main()
