import numpy as np
import math
__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy",'step_lr','long_cosine_lr']


def get_policy(name=None):
    if name is None:
        return constant_lr

    out_dict = {
        "cifar_lr": cifar_lr,
        "constant_lr": constant_lr,
        "val_dependent_lr": val_dependent_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "step_lr": step_lr,
        "long_cosine_lr":long_cosine_lr
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def cifar_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        optim_factor = 0
        #hard coded for tiny Imagenet learning schedule
        if epoch < warmup_length:
            lr = _warmup_lr(lr, warmup_length, epoch)

        if epoch > 160:
            optim_factor = 3
        elif epoch > 120:
            optim_factor = 2
        elif epoch > 60:
            optim_factor = 1
            
        lr = lr*math.pow(0.2, optim_factor)
        
        assign_learning_rate(optimizer, lr)

        return lr
    

    return _lr_adjuster


def constant_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        if epoch < warmup_length:
            lr = _warmup_lr(lr, warmup_length, epoch)

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def val_dependent_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(lr, factor):
        if factor is not None:
            lr = lr / factor
            assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        if epoch < warmup_length:
            lr = _warmup_lr(lr, warmup_length, epoch)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def long_cosine_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        num_generations = kwargs["num_generations"]

        if epoch < warmup_length and gen==0:
            lr = _warmup_lr(lr, warmup_length, epoch)
        else:
            e = epoch + (epochs * gen) - warmup_length
            es = (epochs * num_generations) - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def step_lr(optimizer, warmup_length, **kwargs):
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        num_generations = kwargs["num_generations"]
        #hard coded for tiny Imagenet learning schedule
        if num_generations > 1:
            if epoch < warmup_length:
                lr = _warmup_lr(lr, warmup_length, epoch)

            if epoch >= 80 < 120:
                lr /= 10
            elif epoch >= 120:
                lr /= 100
        else:
            if epoch < warmup_length:
                lr = _warmup_lr(lr, warmup_length, epoch)
            if (80 * (epochs / 160)) <= epoch < (120 * (epochs / 160)):
                lr /= 10
            elif epoch >= (120 * (epochs / 160)):
                lr /= 100

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def multistep_lr(optimizer, warmup_length, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    def _lr_adjuster(epoch, gen, iteration):
        lr = kwargs["lr"]
        lr_adjust = 30
        lr = lr * (0.1 ** (epoch // lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

# TODO: Change the warmup scheduling 
def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
