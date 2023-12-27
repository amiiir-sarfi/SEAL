import os.path as osp
import os
from enum import Enum, auto
import wandb
import torch
import functools
import abc
from datetime import datetime


    
class Stage(Enum):
    """Simple enum to track stage of experiments."""
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path,exist_ok=True)
        
def get_checkpoints_dir(run_info_dir, set, nrlm, nrwm, exp_mode='main'):
    if nrlm == 1 and nrwm == 1:
        nrlm = 'baseline'
    else:
        nrlm = f"mul{nrlm}"
    
    now = datetime.now()
    checkpoints_dir = f'{run_info_dir}/checkpoints/{set}/{now.strftime("%m-%d-%Y_%H-%M-%S")}/{exp_mode}/{nrlm}'
    touch_dir(checkpoints_dir)
    return checkpoints_dir

def summary_logging(cfg, summary, stage ):
    # print ("\033[A\033[A") # Remove the tqdm statement
    out = {Stage.TRAIN:'Training --> ', Stage.VAL:'Val --> ', Stage.TEST:'Testing --> '}
    out = out[stage]
    
    keys = ['Loss', 'Acc1', 'Acc5', 'Best Val Acc1', 'Best Test Acc1', 'lr Reset', 'lr NonReset', 'WD Reset', 'WD NonReset', 'time']
    tmp_out = ""
    for k,v in summary.items():
        k = k.split('/')[-1] 
        if k in keys:
            tmp_out += k + '=' + str(round(v,4)) + ', '
        if k == 'gen':
            gen = v
        if k == 'epoch_in_gen':
            epoch = v
    
    out = out + f"[gen {gen}, epoch {epoch}]: " + tmp_out[:-2]
    
    print(out)

                        
    wandb_logging(cfg, summary)

    # touch_dir(cfg.run_info_dir)
    # TODO: Later do the json dump for plotting and stuff

def wandb_logging(cfg, summary):
    if not cfg.no_wandb and not cfg.debug:
        wandb.log(summary)

def save_checkpoint(cfg, ckpt_dict, file_name='model.pt'):
    ckpt_folder = osp.join(cfg.checkpoints_dir,
    f"lrmul{cfg.ascenders_lr_multiplier}_mulepochs{cfg.ascending_epochs}_max_epochs{cfg.epochs}_ngen{cfg.num_generations}_wdmul{cfg.ascenders_wd_multiplier}_warmup{cfg.warmup_length}_{cfg.layer_threshold}_lr{cfg.lr}_bs{cfg.batch_size}_a{cfg.arch}_mode{cfg.exp_mode}_mom{cfg.momentum}_wd{cfg.weight_decay}_label_smoothing{cfg.label_smoothing}_reverse{cfg.reverse}_cskd{cfg.criterion}_{cfg.lr_policy}_fcascend{cfg.fc_never_ascend}_freeze_descenders{cfg.freeze_descenders}", 
    f"gen{ckpt_dict['gen']}"
    )
    touch_dir(ckpt_folder)
    ckpt_path = osp.join(ckpt_folder, file_name)
    torch.save(ckpt_dict, ckpt_path)

def get_checkpoint_dir(cfg, gen, file_name='model.pt'):
    ckpt_folder = osp.join(cfg.checkpoints_dir,
    f"lrmul{cfg.ascenders_lr_multiplier}_mulepochs{cfg.ascending_epochs}_max_epochs{cfg.epochs}_ngen{cfg.num_generations}_wdmul{cfg.ascenders_wd_multiplier}_warmup{cfg.warmup_length}_{cfg.layer_threshold}_lr{cfg.lr}_bs{cfg.batch_size}_a{cfg.arch}_mode{cfg.exp_mode}_mom{cfg.momentum}_wd{cfg.weight_decay}_label_smoothing{cfg.label_smoothing}_reverse{cfg.reverse}_cskd{cfg.criterion}_{cfg.lr_policy}_fcascend{cfg.fc_never_ascend}_freeze_descenders{cfg.freeze_descenders}", 
    f"gen{gen}"
    )
    ckpt_path = osp.join(ckpt_folder, file_name)
    return ckpt_path

    

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# METERS
class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == '__main__':
    pass
    
    