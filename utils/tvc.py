import copy
import time
import types
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional
from utils import Bar,  AverageMeter, accuracy
import geoopt




class PatchwiseQModule(nn.Module):
    def __init__(self, C, p, T):
        super().__init__()
        self.C, self.p, self.T = C, p, T
        self.d = C * p * p
        Q0 = init_householder_Qs(self.d, T)
        self.Qs = nn.ParameterList([geoopt.ManifoldParameter(Q0[t].clone(), manifold=geoopt.manifolds.Stiefel()) for t in range(T)])

    def transform(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.p, self.p) \.unfold(3, self.p, self.p)
        Hp, Wp = patches.size(2), patches.size(3)
        patches = patches.permute(0,2,3,1,4,5) \.reshape(B*Hp*Wp, -1)
        return patches, Hp, Wp, B, H, W

    def fold(self, patches, Hp, Wp, B, H, W):
        patches = patches.view(B, Hp, Wp, self.C, self.p, self.p)
        return patches.permute(0,3,1,4,2,5).reshape(B, self.C, H, W)

    def forward(self, x):
        patches, Hp, Wp, B, H, W = self.transform(x)
        outs = []
        for t in range(self.T):
            Q_t = self.Qs[t-1]
            xp = patches @ Q_t.T
            outs.append(self.fold(xp, Hp, Wp, B, H, W))
        return outs


def tra(model, dataset, data, time_step, epoch, optimizer, lr_scheduler, scaler, loss_lambda=0.0, attacker=None, writer=None, lr_orth=0.1, gor_lambda=0.1, p=16):

    start_time = time.time()
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    end        = time.time()
    bar        = Bar('Training', max=len(data))

    model.train()
    train_loss = train_acc = train_samples = 0

    frame0, _ = next(iter(data))
    _, C, H0, W0 = frame0.shape
    if not hasattr(model, 'Q'):
        patch_module = PatchwiseQModule(C, p, time_step).cuda()
        model._patch_module = patch_module
        model.Q = patch_module.Qs
        optimizer.add_param_group({'params': list(patch_module.Qs.parameters()),'lr': optimizer.defaults['lr'] * lr_orth})

    for batch_idx, (frame, label) in enumerate(data, 1):
        frame = frame.float().cuda()
        label = label.cuda()
        label_real = label.repeat(time_step)

        if attacker is not None:
            frame = attacker(frame, label)

        optimizer.zero_grad()
        out_all = []
        x_enc_flats = []
        input_frame = frame
        B, C, H, W = input_frame.shape

        for t in range(time_step):
            with amp.autocast():
                x_enc = model._patch_module(input_frame)[t]
                x_flat = x_enc.reshape(frame.size(0), -1)
                x_enc_flats.append(x_flat)
                out = model(x_enc)
                if t == 0:
                    total = out.detach().clone()
                else:
                    total += out.detach().clone()
                out_all.append(out)

        out_all = torch.cat(out_all, dim=0)
        with amp.autocast():
            if loss_lambda > 0.0:
                label_one_hot = torch.zeros_like(out_all).fill_(1.0).to(out_all.device)
                mse_loss = F.mse_loss(out_all, label_one_hot)
                loss = (1 - loss_lambda) * F.cross_entropy(out_all, label_real) + loss_lambda * mse_loss
            else:
                loss = F.cross_entropy(out_all, label_real)


            if gor_lambda > 0.0:
                m_list = []
                for Ft in x_enc_flats:
                    m = Ft.mean(dim=0)
                    m_list.append(m / (m.norm() + 1e-6))

                mean_loss = 0.0
                for i in range(time_step):
                    for j in range(i + 1, time_step):
                        c = torch.dot(m_list[i], m_list[j])
                        mean_loss += c * c

                loss += mean_loss * gor_lambda
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        batch_loss = loss.item()
        train_loss += batch_loss * label.numel()
        prec1, prec5 = accuracy(total, label, topk=(1,5))
        losses.update(batch_loss, frame.size(0))
        top1.update(prec1, frame.size(0))
        top5.update(prec5, frame.size(0))
        train_samples += label.numel()
        train_acc     += (total.argmax(1)==label).float().sum().item()

        functional.reset_net(model)

        batch_time.update(time.time()-end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=batch_idx,
                        size=len(data),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                    )
        bar.next()

    bar.finish()
    train_loss /= train_samples
    train_acc  /= train_samples
    lr_scheduler.step()
    return train_loss, train_acc


def val(model, dataset, data, time_step, epoch, optimizer, lr_scheduler, scaler, loss_lambda=0.0, attacker=None, writer=None):
    start_time = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Validating', max=len(data))

    model.eval()

    val_loss = 0
    val_acc = 0
    val_samples = 0
    batch_idx = 0

    for frame, label in data:
        batch_idx += 1
        frame = frame.float().cuda()
        label = label.cuda()
        label_real = torch.cat([label for _ in range(time_step)], 0)

        if attacker is not None:
            frame = attacker(frame, label)

        out_all = []
        for t in range(time_step):
            input_frame = model._patch_module(frame)[t]

            with torch.no_grad():
                out = model(input_frame)
                if t == 0:
                    total_frame = out.clone().detach()
                else:
                    total_frame += out.clone().detach()
                out_all.append(out)

        out_all = torch.cat(out_all, 0)

        with torch.no_grad():
            if loss_lambda > 0.0:
                label_one_hot = torch.zeros_like(out_all).fill_(1.0).to(out_all.device)
                mse_loss = F.mse_loss(out_all, label_one_hot)
                loss = (1 - loss_lambda) * F.cross_entropy(out_all, label_real) + loss_lambda * mse_loss
            else:
                loss = F.cross_entropy(out_all, label_real)


        batch_loss = loss.item()
        val_loss += loss.item() * label.numel()

        prec1, prec5 = accuracy(total_frame.data, label.data, topk=(1, 5))
        losses.update(batch_loss, input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        val_samples += label.numel()
        val_acc += (total_frame.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
            batch=batch_idx,
            size=len(data),
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
        )
        bar.next()

    bar.finish()
    val_loss /= val_samples
    val_acc /= val_samples
    if writer is not None:
        writer.add_scalar('train_loss', val_loss, epoch)
        writer.add_scalar('train_acc', val_acc, epoch)
    return val_loss, val_acc


def init_householder_Qs(d: int, T: int) -> torch.Tensor:
    I = torch.eye(d)
    Qs = []
    for k in range(T):
        if k == 0:
            Qs.append(I)
        else:
            v = I[:, 0] - I[:, k % d]
            H = I - 2.0 * (v[:, None] @ v[None, :]) / (v @ v)
            Qs.append(H)
    return torch.stack(Qs, dim=0)