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


def canonicalize_encoding(encoding):
    encoding = encoding.lower()
    aliases = {
        'hyper': 'hypergeometric',
        'hyperencoding': 'hypergeometric',
    }
    encoding = aliases.get(encoding, encoding)
    if encoding not in {'stod', 'hypergeometric'}:
        raise ValueError(f'Unsupported encoding: {encoding}')
    return encoding


class HypergeometricRateEncoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, time_step):
        if torch.any((inputs < 0.0) | (inputs > 1.0)):
            raise ValueError('Hypergeometric encoding expects inputs in [0, 1].')

        remaining_mass = inputs * time_step
        encoded = []
        for t in range(1, time_step + 1):
            remaining_slots = time_step - t + 1
            spike_prob = torch.clamp(remaining_mass / remaining_slots, min=0.0, max=1.0)
            spikes = torch.bernoulli(spike_prob)
            remaining_mass = torch.clamp(remaining_mass - spikes, min=0.0, max=time_step - t)
            encoded.append(spikes)

        return torch.stack(encoded, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.mean(dim=0), None


class HypergeometricEncoder(nn.Module):
    def __init__(self, T, mean=None, std=None):
        super().__init__()
        self.T = T
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32) if mean is not None else torch.tensor([], dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32) if std is not None else torch.tensor([], dtype=torch.float32)
        self.register_buffer('mean', mean_tensor, persistent=False)
        self.register_buffer('std', std_tensor, persistent=False)

    def _stats(self, x):
        if self.mean.numel() == 0 or self.std.numel() == 0:
            return None, None
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = self.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return mean, std

    def forward(self, x):
        mean, std = self._stats(x)
        probs = x
        if mean is not None and std is not None:
            probs = probs * std + mean

        probs = torch.clamp(probs, min=0.0, max=1.0)
        spikes = HypergeometricRateEncoding.apply(probs, self.T)

        if mean is not None and std is not None:
            spikes = (spikes - mean.unsqueeze(0)) / std.unsqueeze(0)

        return spikes




class PatchwiseQModule(nn.Module):
    def __init__(self, C, p, T):
        super().__init__()
        self.C, self.p, self.T = C, p, T
        self.d = C * p * p
        Q0 = init_householder_Qs(self.d, T)
        self.Qs = nn.ParameterList([geoopt.ManifoldParameter(Q0[t].clone(), manifold=geoopt.manifolds.Stiefel()) for t in range(T)])

    def transform(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.p, self.p) \
        .unfold(3, self.p, self.p)
        Hp, Wp = patches.size(2), patches.size(3)
        patches = patches.permute(0,2,3,1,4,5) \
        .reshape(B*Hp*Wp, -1)
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
        return torch.stack(outs, dim=0)


def attach_input_encoder(model, encoding, c_in, p, time_step, normalization_mean=None, normalization_std=None):
    encoding = canonicalize_encoding(encoding)

    if hasattr(model, '_patch_module'):
        model.encoding = encoding
        return model._patch_module

    if encoding == 'stod':
        patch_module = PatchwiseQModule(c_in, p, time_step).cuda()
        model.Q = patch_module.Qs
    else:
        patch_module = HypergeometricEncoder(time_step, normalization_mean, normalization_std).cuda()

    model._patch_module = patch_module
    model.encoding = encoding
    return patch_module


def _temporal_classification_loss(logits_per_step, labels, encoding, loss_lambda=0.0):
    if encoding == 'hypergeometric':
        logits = torch.stack(logits_per_step, dim=0).mean(dim=0)
        targets = labels
    else:
        logits = torch.cat(logits_per_step, dim=0)
        targets = labels.repeat(len(logits_per_step))

    if loss_lambda > 0.0:
        label_one_hot = torch.zeros_like(logits).fill_(1.0).to(logits.device)
        mse_loss = F.mse_loss(logits, label_one_hot)
        return (1 - loss_lambda) * F.cross_entropy(logits, targets) + loss_lambda * mse_loss
    return F.cross_entropy(logits, targets)


def tra(model, dataset, data, time_step, epoch, optimizer, lr_scheduler, scaler, loss_lambda=0.0, attacker=None, writer=None, lr_orth=0.1, gor_lambda=0.1, p=16):

    start_time = time.time()
    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    end        = time.time()
    bar        = Bar('Training', max=len(data))

    model.train()
    train_loss = train_acc = train_samples = 0

    for batch_idx, (frame, label) in enumerate(data, 1):
        frame = frame.float().cuda()
        label = label.cuda()

        if attacker is not None:
            frame = attacker(frame, label)

        optimizer.zero_grad()
        out_all = []
        x_enc_flats = []
        encoded_frames = model._patch_module(frame)

        for t in range(time_step):
            with amp.autocast():
                x_enc = encoded_frames[t]
                x_flat = x_enc.reshape(frame.size(0), -1)
                x_enc_flats.append(x_flat)
                out = model(x_enc)
                if t == 0:
                    total = out.detach().clone()
                else:
                    total += out.detach().clone()
                out_all.append(out)

        with amp.autocast():
            loss = _temporal_classification_loss(out_all, label, getattr(model, 'encoding', 'stod'), loss_lambda)

            if gor_lambda > 0.0 and getattr(model, 'encoding', 'stod') == 'stod':
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

        if attacker is not None:
            frame = attacker(frame, label)

        encoded_frames = model._patch_module(frame)
        out_all = []
        for t in range(time_step):
            input_frame = encoded_frames[t]

            with torch.no_grad():
                out = model(input_frame)
                if t == 0:
                    total_frame = out.clone().detach()
                else:
                    total_frame += out.clone().detach()
                out_all.append(out)

        with torch.no_grad():
            loss = _temporal_classification_loss(out_all, label, getattr(model, 'encoding', 'stod'), loss_lambda)

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
