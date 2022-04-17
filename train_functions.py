import os
import time

import torch
import numpy as np
import json

import wandb
import tqdm

from utils import get_loaders, get_setup


# helping function to normal visualisation in Colaboratory
def foo_():
    time.sleep(0.3)


def train_epoch(model, train_dl, criterion, metric, optimizer, scheduler, device):
    model.train()
    loss_sum = 0
    score_sum = 0
    for X, y in tqdm.tqdm(train_dl):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss = loss.item()
        score = metric(output > 0.5, y).mean().item()
        loss_sum += loss
        score_sum += score
    return loss_sum / len(train_dl), score_sum / len(train_dl)


def eval_epoch(model, val_dl, criterion, metric, device):
    model.eval()
    loss_sum = 0
    score_sum = 0
    for X, y in tqdm.tqdm(val_dl):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y).item()
            score = metric(output > 0.5, y).mean().item()
            loss_sum += loss
            score_sum += score
    return loss_sum / len(val_dl), score_sum / len(val_dl)


def run(cfg, use_wandb=True, load_best_model_mode='loss'):
    assert load_best_model_mode in (
    None, 'loss', 'score'), f"load_best_model_mode ({load_best_model_mode}) not in (None, 'loss', 'score')"
    weights_path = os.path.join(cfg.save_folder, 'weights')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        print(f"{weights_path} created successfully")
    save_path = os.path.join(weights_path, cfg.save_name)

    # <<<<< SETUP >>>>>
    train_loader, val_loader = get_loaders(cfg)
    device = torch.device(cfg.device)
    model, optimizer, scheduler, metric, criterion = get_setup(cfg)
    try:
        model.load_state_dict(torch.load(cfg.pretrained))
        print(f'{cfg.pretrained} weights loaded')
    except:
        pass

    # wandb is watching
    if use_wandb:
        wandb.init(project='HDH', config=cfg, name=cfg.save_name)
        wandb.watch(model, log_freq=100)

    best_val_loss = 1e3
    best_val_score = 0
    last_train_loss = 0
    last_val_loss = 1e3
    early_stopping_flag = 0
    best_state_dict = model.state_dict()
    for epoch in range(1, cfg.epochs + 1):
        print(f'Epoch #{epoch}')

        # <<<<< TRAIN >>>>>
        train_loss, train_score = train_epoch(model, train_loader,
                                              criterion, metric,
                                              optimizer, scheduler, device)
        print('      Score    |    Loss')
        print(f'Train: {train_score:.6f} | {train_loss:.6f}')

        # <<<<< EVAL >>>>>
        val_loss, val_score = eval_epoch(model, val_loader,
                                         criterion, metric, device)
        print(f'Val: {val_score:.6f} | {val_loss:.6f}', end='\n\n')
        metrics = {'train_score': train_score,
                   'train_loss': train_loss,
                   'val_score': val_score,
                   'val_loss': val_loss,
                   'lr': scheduler.get_last_lr()[-1]}

        if use_wandb:  # log metrics to wandb
            wandb.log(metrics)

        # saving best weights by loss
        if val_loss < best_val_loss:
            if os.path.exists(save_path + f'-{cfg.criterion}-{best_val_loss}.pth'):
                os.remove(save_path + f'-{cfg.criterion}-{best_val_loss}.pth')
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path + f'-{cfg.criterion}-{val_loss}.pth')
            if load_best_model_mode == 'loss':
                best_state_dict = model.state_dict()

        # saving best weights by score
        if val_score > best_val_score:
            if os.path.exists(save_path + f'-{cfg.metric}-{best_val_score}.pth'):
                os.remove(save_path + f'-{cfg.metric}-{best_val_score}.pth')
            best_val_score = val_score
            torch.save(model.state_dict(), save_path + f'-{cfg.metric}-{val_score}.pth')
            if load_best_model_mode == 'score':
                best_state_dict = model.state_dict()

        cfg.best_weights = [save_path + f'-{cfg.criterion}-{best_val_loss}.pth',
                            save_path + f'-{cfg.metric}-{best_val_score}.pth']
        cfg.save(replace=True)
        # weapon counter over-fitting
        if train_loss < last_train_loss and val_loss > last_val_loss:
            early_stopping_flag += 1
        if early_stopping_flag == cfg.max_early_stopping:
            print('<<< EarlyStopping >>>')
            break

        last_train_loss = train_loss
        last_val_loss = val_loss

    # loading best weights
    model.load_state_dict(best_state_dict)

    if use_wandb:
        wandb.finish()
    return model
