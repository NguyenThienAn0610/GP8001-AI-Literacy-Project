"""PyTorch architectures, search configurations, and deterministic training."""

import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from career_ml import NUMERIC_COLUMNS, PROFILE_COLUMNS, make_preprocessor
from evaluation import neural_scores as scores, validation_key


FAMILIES = ('deep_mlp', 'residual_network', 'field_transformer')


def configurations():
    """Predeclared search, shared optimisation variants across three families."""
    variants = [
        dict(lr=0.0003, dropout=0.1, weight_decay=0.001, batch_size=256,
             scheduler='constant', encoding='binary', capacity='base', smoothing=0.0),
        dict(lr=0.001, dropout=0.1, weight_decay=0.001, batch_size=256,
             scheduler='constant', encoding='binary', capacity='base', smoothing=0.0),
        dict(lr=0.003, dropout=0.1, weight_decay=0.001, batch_size=256,
             scheduler='constant', encoding='binary', capacity='base', smoothing=0.0),
        dict(lr=0.001, dropout=0.2, weight_decay=0.01, batch_size=512,
             scheduler='plateau', encoding='binary', capacity='wide', smoothing=0.0),
        dict(lr=0.0003, dropout=0.3, weight_decay=0.1, batch_size=512,
             scheduler='cosine', encoding='tfidf', capacity='deep', smoothing=0.05),
        dict(lr=0.003, dropout=0.2, weight_decay=0.01, batch_size=512,
             scheduler='plateau', encoding='tfidf', capacity='base', smoothing=0.0),
    ]
    result = []
    for family in FAMILIES:
        for number, values in enumerate(variants, 1):
            cfg = {'id': f'{family}_{number:02d}', 'family': family, **values}
            if family == 'deep_mlp':
                cfg['hidden'] = {'base': [256, 128, 64], 'wide': [512, 256, 128],
                                 'deep': [512, 256, 128, 64, 32]}[cfg['capacity']]
            elif family == 'residual_network':
                cfg.update({'base': dict(width=128, blocks=2),
                            'wide': dict(width=256, blocks=3),
                            'deep': dict(width=128, blocks=4)}[cfg['capacity']])
            else:
                cfg.update({'base': dict(width=32, blocks=2, heads=4),
                            'wide': dict(width=64, blocks=2, heads=4),
                            'deep': dict(width=32, blocks=4, heads=4)}[cfg['capacity']])
            result.append(cfg)
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocessing(encoding):
    pre = make_preprocessor().set_params(profile__tfidf__ngram_range=(1, 1), sparse_threshold=0)
    if encoding == 'binary':
        pre.set_params(profile__tfidf__use_idf=False, profile__tfidf__binary=True,
                       profile__tfidf__norm=None)
    return pre


def feature_groups(pre):
    terms = pre.named_transformers_['profile']['tfidf'].get_feature_names_out()
    groups = [[i for i, term in enumerate(terms) if term.startswith(field.lower() + '=')]
              for field in PROFILE_COLUMNS]
    groups += [[len(terms) + i] for i in range(len(NUMERIC_COLUMNS))]
    assert all(groups)
    assert sorted(i for group in groups for i in group) == list(range(len(terms) + 2))
    return groups


class DeepMLP(nn.Module):
    def __init__(self, inputs, classes, cfg):
        super().__init__()
        layers = []
        for hidden in cfg['hidden']:
            layers += [nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.GELU(),
                       nn.Dropout(cfg['dropout'])]
            inputs = hidden
        self.layers = nn.Sequential(*layers, nn.Linear(inputs, classes))

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.branch = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                    nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.branch(x)


class ResidualNetwork(nn.Module):
    def __init__(self, inputs, classes, cfg):
        super().__init__()
        width = cfg['width']
        self.layers = nn.Sequential(
            nn.Linear(inputs, width),
            *[ResidualBlock(width, cfg['dropout']) for _ in range(cfg['blocks'])],
            nn.LayerNorm(width), nn.GELU(), nn.Linear(width, classes))

    def forward(self, x):
        return self.layers(x)


class FieldTransformer(nn.Module):
    """One learned token per original field, plus a classification token."""
    def __init__(self, groups, classes, cfg):
        super().__init__()
        width = cfg['width']
        self.projections = nn.ModuleList([nn.Linear(len(group), width) for group in groups])
        for index, group in enumerate(groups):
            self.register_buffer(f'indices_{index}', torch.tensor(group, dtype=torch.long))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(width, cfg['heads'], width * 4,
                                            dropout=cfg['dropout'], activation='gelu',
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg['blocks'],
                                               norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        tokens = [projection(x.index_select(1, getattr(self, f'indices_{i}')))
                  for i, projection in enumerate(self.projections)]
        tokens = torch.stack(tokens, dim=1)
        x = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(x)[:, 0])


def make_network(cfg, inputs, groups, classes, device):
    if cfg['family'] == 'deep_mlp':
        net = DeepMLP(inputs, classes, cfg)
    elif cfg['family'] == 'residual_network':
        net = ResidualNetwork(inputs, classes, cfg)
    else:
        net = FieldTransformer(groups, classes, cfg)
    return net.to(device)


@torch.no_grad()
def probabilities(net, x):
    net.eval()
    p = torch.cat([net(batch).softmax(dim=1) for batch in x.split(1024)]).cpu().numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    return p


def train_epoch(net, optimizer, loss_fn, x, y, batch_size):
    net.train()
    order = torch.randperm(len(x), device=x.device)
    total = torch.zeros((), device=x.device)
    for indices in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(net(x[indices]), y[indices])
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.detach() * len(indices)
    value = float((total / len(x)).item())
    if not np.isfinite(value):
        raise RuntimeError('Non-finite training loss')
    return value


def fit_trial(cfg, matrices, y_train, y_valid, classes, device, seed, max_epochs, patience):
    seed_all(seed)
    x, valid_x, groups = matrices
    net = make_network(cfg, x.shape[1], groups, classes, device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['smoothing'])
    schedule = None
    if cfg['scheduler'] == 'plateau':
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
    elif cfg['scheduler'] == 'cosine':
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5)
    history, best, best_epoch = [], None, 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        loss = train_epoch(net, optimizer, loss_fn, x, y_train, cfg['batch_size'])
        metric = scores(y_valid, probabilities(net, valid_x))
        history.append({'epoch': epoch, 'learning_rate': lr, 'training_loss': loss, **metric})
        if best is None or validation_key(metric) > validation_key(best):
            best, best_epoch = metric, epoch
        if schedule is not None:
            if cfg['scheduler'] == 'plateau':
                schedule.step(metric['log_loss'])
            else:
                schedule.step()
        if epoch >= max(10, best_epoch + patience):
            break
    result = {'config_id': cfg['id'], 'family': cfg['family'],
              'best_epoch': best_epoch, 'epochs_run': epoch, 'hit_epoch_limit': epoch == max_epochs,
              'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
              'encoded_features': x.shape[1], 'validation': best,
              'seconds': time.perf_counter() - started, 'history': history}
    del net, optimizer
    return result


def refit(cfg, trial, x, y, groups, classes, device, seed):
    seed_all(seed)
    net = make_network(cfg, x.shape[1], groups, classes, device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['smoothing'])
    # Replay the inner-selected schedule. The outer test fold never controls it.
    for epoch in range(trial['best_epoch']):
        for group in optimizer.param_groups:
            group['lr'] = trial['history'][epoch]['learning_rate']
        train_epoch(net, optimizer, loss_fn, x, y, cfg['batch_size'])
    return net


