import torch.nn as nn
import torch
import math
from .resnet import resnet18, resnet50



def load_ckpt(cfg):
    model = get_model(cfg).cuda()
    ckpt = torch.load(cfg.ckpt_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    start_gen = ckpt['gen']
    start_epoch = ckpt['epoch_in_gen'] + 1

    best_gen_val = best_gen_test = ckpt['gen']
    best_epoch_val = best_epoch_test = ckpt['epoch_in_gen']
    best_test_acc1 = 0
    if cfg.resume_from_next_gen:
        start_gen += 1
        start_epoch = 0
    
    best_val_acc1 = ckpt['best_val_acc']
    best_epoch_val = ckpt['best_epoch_val']
    best_gen_val = ckpt['best_gen_val']
    _best_acc = best_val_acc1 # For printing purposes
    _cur_acc = ckpt['cur_val_acc']
    if 'best_test_acc' in ckpt.keys():
        best_test_acc1 = ckpt['best_test_acc']
        best_epoch_test = ckpt['best_epoch_test']
        best_gen_test = ckpt['best_gen_test']
        _best_acc = best_test_acc1
        _cur_acc = ckpt['cur_test_acc']

    print(f"checkpoint loaded at gen {ckpt['gen']} epoch {ckpt['epoch']} with best accuracy {_best_acc} and current accuracy {_cur_acc}")
    print(f"Restarting training from gen {start_gen} and epoch {start_epoch}")
    # initial wandb logging
    ckpt_summary = {
            'Val/Acc1':ckpt['best_val_acc'],
            'Best/Best Val Acc1': ckpt['best_val_acc'],
            'Train/gen':ckpt['gen'],
            'Train/epoch_in_gen': ckpt['epoch_in_gen'],
            'Train/epoch':  ckpt['epoch'], # Useful for wandb
            }
    if 'best_test_acc' in ckpt:
        ckpt_summary['Test Acc1'] = ckpt['best_test_acc']
        ckpt_summary['Best Test Acc1'] = ckpt['best_test_acc']
    
    return (
        model, start_gen, start_epoch,
        best_val_acc1, best_epoch_val, best_gen_val,
        best_test_acc1, best_epoch_test, best_gen_test,
        ckpt_summary, ckpt
    )



def _reinit(real_weights):
    new_weights = torch.zeros_like(real_weights).cuda()
    nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
    real_weights.data = new_weights
    
def reinitalize_forgetting(model, cfg):
    """Reinitilizes the forgetting hypothesis, and seperates fit and forgetting hypothesese for the purpose of an ablation study

    Returns:
        a tuple of (fit hypothesis, forgetting hypothesis). 
        All of which are dictionaries of {layer_name:tensor}.
    """
    layer_threshold = cfg.layer_threshold
    print("REINITIALIZING FORGETTING")

    print('reverse', cfg.reverse)
    layer_threshold_reached = False
    descenders_list = {}
    ascenders_list = {}
    for n, m in  model.named_parameters():
        dots_in_reset_layer = layer_threshold.count('.')
        
        # is this a norm layer?
        if ('norm' in n.lower() or 'bn' in n.lower()):
            if layer_threshold_reached:
                descenders_list[n] = m
            else:
                ascenders_list[n] = m
            continue

        # Check if we have reached the reset layer
        if not layer_threshold_reached:
            if '.'.join(n.split('.')[:dots_in_reset_layer+1]) == layer_threshold:
                layer_threshold_reached = True
            else:
                ascenders_list[n] = m
                if cfg.reverse:
                    if 'conv' in n or 'downsample.0' in n:
                        _reinit(m)
                    
                        # Resetting conv biases
                        if 'bias' in n and m is not None:
                            rand_tensor = torch.zeros_like(m).cuda()
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(rand_tensor, -bound, bound)
                            m.data = rand_tensor
                        # Resetting linear
                        if 'fc' in n or 'linear' in n:
                            _reinit(m)
                            if 'bias' in n and m is not None and False:
                                rand_tensor = torch.zeros_like(m).cuda()
                                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m)
                                bound = 1 / math.sqrt(fan_in)
                                nn.init.uniform_(rand_tensor, -bound, bound)
                                m.data = rand_tensor
                    
                    print("newW", n)
                continue
                
        if layer_threshold_reached:
            if 'weight' in n and m is not None:         
                if cfg.fc_never_ascend and ('fc' in n or 'linear' in n) and cfg.reverse:
                    ascenders_list[n] = m
                    continue

                descenders_list[n] = m
                # Resetting conv weights
                if not cfg.reverse:
                    if 'conv' in n or 'downsample.0' in n:
                        _reinit(m)
                    
                        # Resetting conv biases
                        if 'bias' in n and m is not None:
                            rand_tensor = torch.zeros_like(m).cuda()
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(rand_tensor, -bound, bound)
                            m.data = rand_tensor
                    # Resetting linear
                    if 'fc' in n or 'linear' in n:
                        _reinit(m)
                        if 'bias' in n and m is not None and False:
                            rand_tensor = torch.zeros_like(m).cuda()
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(rand_tensor, -bound, bound)
                            m.data = rand_tensor

                    print("newW", n)

    if cfg.freeze_descenders:
        descenders_list = {}

    if cfg.reverse:
        return descenders_list, ascenders_list
    return ascenders_list, descenders_list


def get_ascenders_descenders(model, cfg, ascending):
    """ 
    Just returns lists of ascender and descender layers without reinitializing.

    Returns:
        a tuple of (fit hypothesis, forgetting hypothesis). 
        All of which are dictionaries of {layer_name:tensor}.
    """
    print("GETTING ASCENDERS AND DESCENDERS")

    layer_threshold = cfg.layer_threshold
    layer_threshold_reached = False
    descenders_list = {}
    ascenders_list = {}
    for n, m in  model.named_parameters():
        dots_in_reset_layer = layer_threshold.count('.')
        

        if ('bn' in n.lower() or 'downsample.1' in n.lower() or 'norm' in n.lower()):
            if layer_threshold_reached:
                descenders_list[n] = m
            else:
                ascenders_list[n] = m
            continue

        # Check if we have reached the reset layer
        if not layer_threshold_reached:
            if '.'.join(n.split('.')[:dots_in_reset_layer+1]) == layer_threshold:
                layer_threshold_reached = True
            else:
                ascenders_list[n] = m
                continue
                
        if layer_threshold_reached:
            if cfg.fc_never_ascend and ('fc' in n or 'linear' in n) and cfg.reverse:
                ascenders_list[n] = m
                continue
                
            descenders_list[n] = m
    
    # For an ablation study, only during ascending phase, we froze the fit hypothesis
    if cfg.freeze_descenders and ascending:
        descenders_list = {}
        
    if cfg.reverse:
        return descenders_list, ascenders_list
    return ascenders_list, descenders_list




def get_model(cfg):
    if cfg.arch == 'ResNet18':
        return resnet18(num_classes=cfg.num_cls)
    
    if cfg.arch == 'ResNet50':
        return resnet50(num_classes=cfg.num_cls)

if __name__ == "__main__":
    from types import SimpleNamespace
    cfg_resnet = SimpleNamespace(arch='ResNet50', num_cls=200, fc_never_ascend=True, reverse=0, freeze_descenders=False,layer_threshold= 'layer3')
    
    resnet = get_model(cfg_resnet)
    # for idx, (k, v) in enumerate(model.named_parameters()):
    #     print(idx, k)
    
    print("RINITIALIZE FORGETTING:")
    print(f"RESNET with reverse {cfg_resnet.reverse}")
    ascenders, descenders = reinitalize_forgetting(resnet, cfg_resnet)
        
    for k in descenders:
            print('descenders', k)
    for k in ascenders:
            print('ascenders', k)

    print()
    print('-'*80)
    print()

    print("NO REINITIALIZATION:")
    print(f"RESNET with reverse {cfg_resnet.reverse}")
    ascenders, descenders = get_ascenders_descenders(resnet, cfg_resnet, ascending=True)

    for k in descenders:
            print('descenders', k)
    for k in ascenders:
            print('ascenders', k)

    print()
    print('-'*80)
    print()
        