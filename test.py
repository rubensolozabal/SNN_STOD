import os

import torch

from utils import numpy_compat
from utils.config import get_args, get_data, get_net
from utils.tvc import attach_input_encoder, val


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _infer_patch_size_from_q_shape(state_dict, c_in):
    q_key = next((key for key in ('_patch_module.Qs.0', 'Q.0') if key in state_dict), None)
    if q_key is None:
        return None

    q_tensor = state_dict[q_key]
    if q_tensor.ndim != 2 or q_tensor.shape[0] != q_tensor.shape[1]:
        return None

    flat_patch_dim = q_tensor.shape[0]
    if flat_patch_dim % c_in != 0:
        return None

    patch_area = flat_patch_dim // c_in
    patch_size = int(round(patch_area ** 0.5))
    if patch_size * patch_size != patch_area:
        return None
    return patch_size


def _format_shape_mismatch_message(model, state_dict, checkpoint_path):
    model_state = model.state_dict()
    mismatched_shapes = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            mismatched_shapes.append((key, tuple(model_state[key].shape), tuple(value.shape)))

    if not mismatched_shapes:
        return None

    checkpoint_patch_size = _infer_patch_size_from_q_shape(state_dict, c_in=3)
    model_patch_size = _infer_patch_size_from_q_shape(model_state, c_in=3)
    lines = [f'Checkpoint shape mismatch while loading {checkpoint_path}:']
    for key, model_shape, checkpoint_shape in mismatched_shapes[:8]:
        lines.append(f'  {key}: model expects {model_shape}, checkpoint has {checkpoint_shape}')
    if len(mismatched_shapes) > 8:
        lines.append(f'  ... and {len(mismatched_shapes) - 8} more mismatched tensors')

    if checkpoint_patch_size is not None and model_patch_size is not None and checkpoint_patch_size != model_patch_size:
        lines.append(
            f'This STOD checkpoint appears to have been trained with -p {checkpoint_patch_size}, '
            f'but the current evaluation model was built with -p {model_patch_size}.'
        )
        lines.append(f'Re-run evaluation with -p {checkpoint_patch_size} (or use a checkpoint trained with -p {model_patch_size}).')

    return '\n'.join(lines)


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


def load_model_weights(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('net') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    shape_mismatch_message = _format_shape_mismatch_message(model, state_dict, checkpoint_path)
    if shape_mismatch_message is not None:
        raise RuntimeError(shape_mismatch_message)

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

    _, test_data_loader, c_in, num_classes, normalization_mean, normalization_std = get_data(
        args.b,
        args.j,
        args.T,
        args.data_dir,
        args.dataset,
        args.imagenet_backend,
        args.imagenet_hf_path,
        args.imagenet_hf_name,
        args.imagenet_hf_train_split,
        args.imagenet_hf_val_split,
        args.imagenet_hf_cache_dir,
    )
    net = get_net(args.surrogate, args.dataset, args.model, num_classes, args.drop_rate, args.tau, c_in)
    attach_input_encoder(net, args.encoding, c_in, args.p, args.T, normalization_mean, normalization_std)

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
