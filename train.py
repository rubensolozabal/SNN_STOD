import datetime
from utils import numpy_compat
from torch.utils.tensorboard import SummaryWriter
from utils.config import *
from utils.tvc import *
import collections
import torchattacks
import geoopt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def main():
    args = get_args()
    train_data_loader, test_data_loader, c_in, num_classes, normalization_mean, normalization_std = get_data(
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
    if args.attack == 'fgsm':
        attacker = torchattacks.FGSM(net, eps=args.eps / 255)
    elif args.attack == 'pgd':
        attacker = torchattacks.PGD(net, eps=args.eps / 255, alpha=args.alpha, steps=args.steps)
    else:
        attacker = None

    # optimizer preparing
    if net.encoding == 'stod':
        orth_params = list(net._patch_module.Qs.parameters())
        orth_param_ids = {id(param) for param in orth_params}
        base_params = [param for param in net.parameters() if id(param) not in orth_param_ids]
        param_groups = [
            {'params': base_params, 'momentum': args.momentum, 'weight_decay': args.weight_decay},
            {'params': orth_params, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'lr': args.lr * args.lr_orth},
        ]
    else:
        param_groups = [{'params': net.parameters(), 'momentum': args.momentum, 'weight_decay': args.weight_decay}]
    optimizer = geoopt.optim.RiemannianSGD(param_groups, lr=args.lr)


    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # loading models from checkpoint
    start_epoch = 0
    max_val_acc = 0

    # TODO: if args.resume:
    # this calls for a design for Q-matrix loading

    out_dir = os.path.join(args.out_dir, f'{args.dataset}_{args.model}_enc{net.encoding}_T{args.T}_or{args.lr_orth}_tau{args.tau}_e{args.epochs}_bs{args.b}_wd{args.weight_decay}_reg{args.gor_lambda}')

    if args.attack is not None:
        if args.attack == 'fgsm':
            out_dir += f'_fgsm_eps{args.eps}'
        elif args.attack == 'pgd':
            out_dir += f'_pgd_eps{args.eps}_alpha{args.alpha}_steps{args.steps}'
        else:
            out_dir += f'_clean'

    if net.encoding == 'stod' and args.p is not None:
        out_dir += f'_p{args.p}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Make Dir {out_dir}.')
    else:
        print('out Dir already exists:', out_dir)

    # save the initialization of parameters
    if args.save_init:
        checkpoint = {
            'net': net.state_dict(),
            'epoch': 0,
            'max_val_acc': 0.0
        }
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_0.pth'))
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    # training and validating
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = tra(model=net, dataset=args.dataset, data=train_data_loader, time_step=args.T, epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, loss_lambda=args.loss_lambda, attacker=attacker, writer=writer, lr_orth=args.lr_orth, gor_lambda=args.gor_lambda, p=args.p)
        val_loss, val_acc = val(model=net, dataset=args.dataset, data=test_data_loader, time_step=args.T, epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, loss_lambda=args.loss_lambda, attacker=attacker, writer=writer)

        save_max = False
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_val_acc': max_val_acc,
        }
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, max_val_acc={max_val_acc:.4f}, total_time={total_time:.4f}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')
        if epoch == 0:
            print("Memory Reserved: %.4fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

if __name__ == '__main__':
    main()
