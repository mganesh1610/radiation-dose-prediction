"""Deterministic, source-balanced A–D neural objectives for development experiments."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn


class DoseNetwork(nn.Module):
    def __init__(self, features, hidden=(64, 16)):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, hidden[0]), nn.ReLU(),
                                     nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.dose = nn.Linear(hidden[1], 1)
        self.exposure = nn.Linear(hidden[1], 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.dose(h).flatten(), self.exposure(h).flatten()


def balanced_weights(studies, donors, classes=None):
    """Equal study -> (optional class) -> donor -> sample; sum exactly one."""
    studies, donors = np.asarray(studies), np.asarray(donors)
    if studies.ndim != 1 or studies.shape != donors.shape or not len(studies):
        raise ValueError('Invalid weighting axes')
    if any(len(set(studies[donors == d])) != 1 for d in set(donors)):
        raise ValueError('Donor spans studies')
    if classes is not None:
        classes = np.asarray(classes)
        if classes.shape != studies.shape or not set(classes).issubset({0, 1}):
            raise ValueError('Invalid exposure labels')
    w = np.zeros(len(studies))
    for s in sorted(set(studies)):
        s_mask = studies == s
        levels = [None] if classes is None else sorted(set(classes[s_mask]))
        if classes is not None and levels != [0, 1]:
            raise ValueError('Both exposure classes required per study')
        for level in levels:
            mask = s_mask if level is None else s_mask & (classes == level)
            groups = sorted(set(donors[mask]))
            for d in groups:
                take = mask & (donors == d)
                w[take] = 1 / (len(set(studies)) * len(levels) * len(groups) * take.sum())
    if not np.isclose(w.sum(), 1) or np.any(w <= 0):
        raise ValueError('Invalid source weights')
    return w


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cpu')


def objective(model, batch, strategy, params, pretrain=False):
    """Auxiliary numerical labels are not read by B or C, including NaN sentinels."""
    if pretrain:
        _, logits = model(batch['aux_x'])
        return (nn.functional.binary_cross_entropy_with_logits(logits, batch['aux_exposure'], reduction='none')
                * batch['aux_binary_weights']).sum()
    pred, logits = model(batch['core_x'])
    core_loss = ((pred - batch['core_y']) ** 2 * batch['core_weights']).sum()
    if strategy in ('A', 'B'):
        return core_loss
    aux_pred, aux_logits = model(batch['aux_x'])
    if strategy == 'C':
        core_binary = (nn.functional.binary_cross_entropy_with_logits(logits, batch['core_exposure'], reduction='none')
                       * batch['core_binary_weights']).sum()
        aux_binary = (nn.functional.binary_cross_entropy_with_logits(aux_logits, batch['aux_exposure'], reduction='none')
                      * batch['aux_binary_weights']).sum()
        return core_loss + params['classification_lambda'] * .5 * (core_binary + aux_binary)
    if strategy == 'D':
        aux_loss = ((aux_pred - batch['aux_y']) ** 2 * batch['aux_weights']).sum()
        return (1 - params['auxiliary_mass']) * core_loss + params['auxiliary_mass'] * aux_loss
    raise ValueError('Unknown strategy')


def fit_network(batch, params, seed, hidden=(64, 16)):
    if params['strategy'] not in 'ABCD' or params['steps'] < 1:
        raise ValueError('Invalid training plan')
    for key in ('core_x', 'core_y', 'core_weights'):
        if not torch.isfinite(batch[key]).all():
            raise ValueError('Nonfinite training input')
    if params['strategy'] != 'A' and not torch.isfinite(batch['aux_x']).all():
        raise ValueError('Nonfinite auxiliary expression')
    if params['strategy'] == 'D' and not torch.isfinite(batch['aux_y']).all():
        raise ValueError('D requires verified numerical auxiliary doses')
    torch.manual_seed(seed)
    model = DoseNetwork(batch['core_x'].shape[1], hidden)
    started = time.perf_counter(); history = []
    phases = [('pretrain', params['pretrain_steps'])] if params['strategy'] == 'B' else []
    phases.append(('fit', params['steps']))
    for phase, steps in phases:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'], foreach=False)
        first = None
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, batch, params['strategy'], params, phase == 'pretrain')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if first is None: first = float(loss.detach())
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
        with torch.no_grad():
            last = float(objective(model, batch, params['strategy'], params, phase == 'pretrain'))
        if not np.isfinite(last): raise ValueError('Nonfinite final loss')
        history.append(dict(phase=phase, steps=steps, initial_loss=first, final_loss=last))
    model.eval()
    return model, dict(seconds=time.perf_counter()-started, seed=seed, phases=history,
                       parameters=sum(p.numel() for p in model.parameters()))


def predict(model, x, target_mean, target_scale):
    model.eval()
    with torch.no_grad(): result = model(tensor(x))[0].numpy().astype(float) * target_scale + target_mean
    if not np.isfinite(result).all(): raise ValueError('Nonfinite prediction')
    return result


def save_model(path, model, params, seed, target_mean, target_scale, hidden=(64, 16)):
    path = Path(path); temporary = path.with_suffix('.partial')
    torch.save(dict(state=model.state_dict(), features=model.encoder[0].in_features, hidden=list(hidden),
                    params=params, seed=int(seed), target_mean=float(target_mean), target_scale=float(target_scale)), temporary)
    temporary.replace(path)


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = DoseNetwork(saved['features'], saved['hidden']); model.load_state_dict(saved['state']); model.eval()
    return model, saved
