import time
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tqdm.contrib import tqdm
from models.model_utils import get_model, reinitalize_forgetting, get_ascenders_descenders
from data.datasets import load_dataset
from utils import (
    Stage, 
    save_checkpoint, 
    summary_logging, 
    rgetattr, 
    wandb_logging, 
    AverageMeter, 
    accuracy, 
    get_optimizer, 
    get_policy, 
    KDLoss, 
    LabelSmoothing
)

from models.model_utils import load_ckpt

class IterativeTrainer:
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        if self.cfg.arch in ["ResNet18", "ResNet50"]:
            self.track_gradient_list = [
                'conv1.weight', 
                'layer1.0.conv1.weight',
                'layer2.1.conv1.weight',
                'layer3.1.conv1.weight',
                'layer4.1.conv1.weight', 
                'fc.weight'
            ]
        else:
            self.track_gradient_list = None

        if cfg.auto_mix_prec and False:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Criterion
        if cfg.label_smoothing == 0:
            softmax_criterion = nn.CrossEntropyLoss().cuda()
        else:
            softmax_criterion = LabelSmoothing(smoothing=cfg.label_smoothing).cuda()
        
        self.criterion = lambda output,target: softmax_criterion(output, target)
        
        if cfg.criterion == "cs_kd":
            self.kdloss = KDLoss(4).cuda()
        
        # Model
        if model == None:
            self.model = get_model(cfg).cuda()
        else:
            self.model = model.cuda()
        # Overall Best:
        self.best_val_acc1 = -1
        self.best_test_acc1 = -1
        self.best_gen_val = -1
        self.best_gen_test = -1
        self.best_epoch_val = -1
        self.best_epoch_test = -1
        self.ascenders_sign = +1
        
        self.start_gen = 0
        self.start_epoch = 0
        
        # If we are resuming from the middle of a generation
        self.resume = False
        
        # value of the last nrlme epoch
        self.ascent_over_epoch = cfg.ascending_epochs

        # Loading checkpoint
        if self.cfg.ckpt_dir:
            (
                self.model, self.start_gen, self.start_epoch,
                self.best_val_acc1, self.best_epoch_val, self.best_gen_val,
                self.best_test_acc1, self.best_epoch_test, self.best_gen_test,
                ckpt_summary, ckpt
            ) = load_ckpt(cfg)
            
            self.resume = not cfg.resume_from_next_gen # resume from this gen?
            wandb_logging(cfg, ckpt_summary)

        # Optimizer
        self.optimizer = get_optimizer(cfg.optimizer, self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        self._first_grad_dict = True

    def _update_optimizer(self, ascenders=None, descenders=None, gen=0, epoch=0):            
        # rename parameters
        cfg = self.cfg
        _alm, _awdm = cfg.ascenders_lr_multiplier, cfg.ascenders_wd_multiplier 
        print(f'[+] len ascenders {len(ascenders)}, len descenders {len(descenders)} in _update_optimzier')
        # Is fortuitous?
        is_fortuitous = (gen == 0) or (_alm == 1 and _awdm == 1)
        
        if is_fortuitous: # For baseline run (fortuitous forgetting)
            self.optimizer = get_optimizer(cfg.optimizer, self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
            
        else: # group[0] = descenders | group[1] = ascenders
            # If we want to ascend on all, give dummy value to descenders dict to avoid errors
            if len(descenders.values()) == 0: 
                descenders = {"dummy": torch.tensor([2.34], requires_grad=True)}
                
            self.optimizer = get_optimizer(cfg.optimizer, descenders.values(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
            self.optimizer.add_param_group({'params':ascenders.values()})
            
            
            
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
    
        
    def update_summary(self, summary:dict, gen:int) -> dict:
        """Only called for train summary to add some stuff for ablation/additional experiments

        Args:
            summary (dict): summary dictionary

        Returns:
            dict: new train summary dict
        """
        return summary

    @torch.no_grad()
    def on_gen_start(self, gen):
        # If we are resuming from a checkpoint, it is not start of a generation!
        if self.resume:
            self.resume = False
            return
        cfg = self.cfg
        if gen > 0:
            if "fortuitous" in cfg.exp_mode:
                # In case of fortuitous, Ascenders=NonReset, and Descenders=Reset parameters
                ascenders, descenders = reinitalize_forgetting(self.model, cfg)
            
            if "ascend" in cfg.exp_mode:
                ascenders, descenders = get_ascenders_descenders(self.model, cfg, ascending=True)

            if  "normal" in cfg.exp_mode:
                ascenders = {}
                descenders = {'all_params':self.model.parameters()}

            self._update_optimizer(ascenders, descenders, gen=gen)

    def _is_normal(self, gen, epoch):
        """ Sets the behavior of this epoch & gen. 
            if it mimics the baseline: normal (True)
            if it is different from the baseline: not normal (False)?
            If it was NOT normal and from this epoch it's gonna be normal: is_on_change=True

        Args:
            gen (_type_): _description_
            epoch (_type_): _description_

        Returns:
            None. Only sets self.is_normal, self.is_on_change
        """
        cfg = self.cfg
        _alm, _alme = cfg.ascenders_lr_multiplier, cfg.ascending_epochs
        _awdm = cfg.ascenders_wd_multiplier 
        is_fortuitous = (gen == 0) or (_alm == 1 and _awdm == 1) or (_alme <= 0)
        self.is_normal = is_fortuitous or (epoch > _alme) 
        self.is_on_change = (epoch == _alme) and not is_fortuitous
        if self.is_on_change:
            self.ascenders_sign = +1

    def set_lr(self, gen, epoch):
        # rename parameters
        _alm, _alme = self.cfg.ascenders_lr_multiplier, self.cfg.ascending_epochs
        _awdm = self.cfg.ascenders_wd_multiplier 
    
        # In normal behavior epochs?
        if self.is_normal:
            self.lr_scheduler(epoch, gen=gen, iteration=None)
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]['weight_decay'] = self.optimizer.param_groups[0]['weight_decay'] 
                self.optimizer.param_groups[1]['lr'] = self.optimizer.param_groups[0]['lr'] 
            return
        
        # Is it the epoch we change the behavior? FROM ASCEND TO DESCEND
        if self.is_on_change:
            print("[+] Optimizer has been reset due to IS ON CHANGE in _is_normal")
            ascenders, descenders = get_ascenders_descenders(self.model, self.cfg, ascending=False) 

            self._update_optimizer(ascenders, descenders, gen=gen)
            self.lr_scheduler(epoch, gen=gen, iteration=None)
            
            # Everything should ascend:
            self.optimizer.param_groups[1]['weight_decay'] = self.optimizer.param_groups[0]['weight_decay'] 
            self.optimizer.param_groups[1]['lr'] = self.optimizer.param_groups[0]['lr'] 
            return
            
        # This is not the normal behavior: (ascenders have different behavior than descenders)
        # group[0] = descenders and group[1] = ascenders
        self.lr_scheduler(epoch, gen=gen, iteration=None)
        descenders_lr = self.optimizer.param_groups[0]['lr']
        ascenders_lr = descenders_lr * _alm

        if _alm != 1:
            self.optimizer.param_groups[1]['lr'] = ascenders_lr
            
        if _awdm != 1:
            self.optimizer.param_groups[1]['weight_decay'] = _awdm * self.optimizer.param_groups[0]['weight_decay'] 

        
    def compute_forward(self, inputs, targets, stage):
        return self.model(inputs)
    
    def cskd_forward(self, inputs, targets, stage):
        batch_size = inputs.size(0)
        loss_batch_size = batch_size // 2
        targets_ = targets[:batch_size // 2]
        outputs = self.model(inputs[:batch_size // 2])
        loss = self.criterion(outputs, targets_)
        
        with torch.no_grad():
            outputs_cls = self.model(inputs[batch_size // 2:])
            
        cls_loss = self.kdloss(outputs[:batch_size // 2], outputs_cls.detach())
        lamda = 3
        loss += lamda * cls_loss
        if outputs.size(-1) >= 5:
            acc1, acc5 = accuracy(outputs, targets_, topk=(1, 5))
        else:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = torch.zeros_like(acc1)
        
        return loss, acc1, acc5
    
    def ce_forward(self, inputs, targets, stage):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        if outputs.size(-1) >= 5:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        else:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = torch.zeros_like(acc1)
        
        return loss, acc1, acc5
            
    # we need self.criterion, self.model, and self.kdloss for this
    def compute_objectives(self, inputs, targets, stage): # returns loss
        """Need self.criterion and self.model to do certain behavior for this function
        """
        if self.cfg.criterion == "cs_kd" and stage == Stage.TRAIN:
            loss, acc1, acc5 = self.cskd_forward(inputs, targets, stage)
        else:
            loss, acc1, acc5 = self.ce_forward(inputs, targets, stage)
        
        self.losses.update(loss.item(), inputs.size(0))
        self.top1.update(acc1.item(), inputs.size(0))
        self.top5.update(acc5.item(), inputs.size(0))
        
        return loss
    
    def fit_batch(self, inputs, outputs):
        """Fit one batch, override to do multiple updates.
        The default implementation depends on a few methods being defined
        with a particular behavior:
        * ``compute_objectives()``
        Also depends on having optimizers passed at initialization.
        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        Returns
        -------
        detached loss
        num of correct predictions
        """
        # Managing automatic mixed precision
        if self.cfg.auto_mix_prec and False:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = self.compute_objectives(inputs, outputs, Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_objectives(inputs, outputs, Stage.TRAIN)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu()
    
    def train_loop(self, train_loader, progressbar, epoch, gen=0):        
        _train_start_time = time.time()

        with tqdm(
            train_loader,
            disable=not progressbar
            ) as t:
                for batch_idx, (inputs, targets) in enumerate(t):
                    inputs, targets = inputs.cuda(),targets.long().squeeze().cuda()
                    loss = self.fit_batch(inputs, targets)
                    
                    self.grad_norms_logging(batch_idx)
                    t.set_postfix(train_loss=self.losses.avg)
                    if self.cfg.debug and batch_idx == self.cfg.debug_batches - 1:
                        break
        
        # ON train end                
        lr_non_reset = self.optimizer.param_groups[0]['lr']
        wd_non_reset = self.optimizer.param_groups[0]['weight_decay']
        if len(self.optimizer.param_groups) > 1:
            lr_non_reset = self.optimizer.param_groups[1]['lr']
            wd_non_reset = self.optimizer.param_groups[1]['weight_decay']
            
        train_summary = {
                'Train/Loss':self.losses.avg,
                'Train/Acc1':self.top1.avg,
                'Train/Acc5':self.top5.avg,
                'LR/lr Reset': self.optimizer.param_groups[0]['lr'] if not self.cfg.reverse else lr_non_reset,
                'LR/lr NonReset': lr_non_reset if not self.cfg.reverse else self.optimizer.param_groups[0]['lr'],
                'LR/WD Reset': self.optimizer.param_groups[0]['weight_decay'] if not self.cfg.reverse else wd_non_reset,
                'LR/WD NonReset': wd_non_reset if not self.cfg.reverse else self.optimizer.param_groups[0]['weight_decay'],
                'Train/gen': gen,
                'Train/epoch_in_gen': epoch,
                'Train/epoch':  epoch + (gen*self.cfg.epochs), # Useful for wandb
                'Train/time': time.time() - _train_start_time
                }
        train_summary = self.update_summary(train_summary, gen)
        summary_logging(self.cfg, train_summary, Stage.TRAIN)
        self._flush_grad_norms(epoch=epoch + (gen*self.cfg.epochs))

    def eval_loop(self, loader):
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

        self.model.eval()

        with torch.no_grad():
            with tqdm(
            loader,
            disable=True,
            ) as t:
                for batch_idx, (inputs, targets) in enumerate(t):
                    inputs, targets = inputs.cuda(),targets.long().squeeze().cuda()
                    _ = self.compute_objectives(inputs, targets, Stage.VAL)
                    t.set_postfix(test_loss=self.top1.avg)
                    if self.cfg.debug and batch_idx == self.cfg.debug_batches - 1:
                        break

    def fit(
        self,
        train_loader=None,
        valid_loader=None,
        progressbar=True
    ):
        train_loader, valid_loader, test_loader = load_dataset(self.cfg)
        for gen in range(self.start_gen, self.cfg.num_generations):
            if self.cfg.debug and gen > self.cfg.debug_gens and self.cfg.debug_gens>0:
                break
            # On generation start
            self.on_gen_start(gen)
            self.gen_best_val_acc = 0
            self.gen_best_test_acc = 0
            bad_val_counter = 0
            
            for epoch in range(self.start_epoch, self.cfg.epochs):
                self.start_epoch = 0
                # ON train start
                self.model.train()
                self.losses = AverageMeter("Loss", ":.4f")
                self.top1 = AverageMeter("Acc@1", ":6.4f")
                self.top5 = AverageMeter("Acc@5", ":6.4f")
                
                self._is_normal(gen, epoch)
                self.set_lr(gen, epoch)
                
                if self.cfg.debug and ((epoch > self.cfg.debug_epochs and self.cfg.debug_epochs>0) or (gen==0 and epoch==2)):
                    break                
                
                self.train_loop(train_loader,progressbar, epoch, gen)

                
                # ON Val start
                if (epoch+1) % self.cfg.val_interval == 0 or epoch == self.ascent_over_epoch or \
                    epoch == self.ascent_over_epoch + self.cfg.save_after_k_epochs_ascentOver:
                    

                    _val_start_time = time.time()
                    self.eval_loop(valid_loader)
                    # ON Val end
                    cur_val_loss = self.losses.avg
                    cur_val_acc1 = self.top1.avg
                    cur_val_acc5 = self.top5.avg

                    # Best validation acc1 in current generation is at this epoch? 
                    is_best_val_gen = cur_val_acc1 > self.gen_best_val_acc
                    self.gen_best_val_acc = max(self.gen_best_val_acc, cur_val_acc1)
                    
                    if cur_val_acc1 * 0.999 < self.gen_best_val_acc: 
                        bad_val_counter += 1
                        
                    is_best_overall = cur_val_acc1 > self.best_val_acc1
                    if is_best_overall:
                        self.best_val_acc1 = cur_val_acc1
                        self.best_gen_val = gen
                        self.best_epoch_val = epoch

                    # save
                    checkpoint_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'cur_val_acc': cur_val_acc1,
                    'best_val_acc': self.best_val_acc1, 
                    'gen_best_val_acc': self.gen_best_val_acc,
                    "optimizer": self.optimizer.state_dict(),
                    'best_epoch_val': self.best_epoch_val,
                    'best_gen_val': self.best_gen_val,
                    'gen':gen,
                    'epoch_in_gen': epoch,
                    'epoch': epoch + (gen*self.cfg.epochs)
                    }
                
                    val_summary = {
                        'Val/Loss':cur_val_loss,
                        'Val/Acc1':cur_val_acc1,
                        'Val/Acc5':cur_val_acc5,
                        'Best/Best Val Acc1': self.best_val_acc1,
                        'Val/gen':gen,
                        'Best/gen':gen,
                        'Val/epoch_in_gen': epoch,
                        'Val/epoch': epoch + (gen*self.cfg.epochs),
                        'Val/time': time.time() - _val_start_time
                        }
                    summary_logging(self.cfg, val_summary, Stage.VAL)

                    # Testing
                    if test_loader is not None:
                        # On test begin
                        self.eval_loop(test_loader)
                    
                        # ON Test end
                        cur_test_loss = self.losses.avg
                        cur_test_acc1 = self.top1.avg
                        cur_test_acc5 = self.top5.avg

                        is_best_test_gen = cur_test_acc1 > self.gen_best_test_acc
                        self.gen_best_test_acc = max(self.gen_best_test_acc, cur_test_acc1)
                        is_best_overall = cur_test_acc1 > self.best_test_acc1
                        if is_best_overall:
                            self.best_test_acc1 = cur_test_acc1
                            self.best_gen_test = gen
                            self.best_epoch_test = epoch

                        # add test info to checkpoint and log test info
                        checkpoint_dict['gen_best_test_acc'] = self.gen_best_test_acc
                        checkpoint_dict['best_test_acc'] = self.best_test_acc1
                        checkpoint_dict['best_epoch_test'] = self.best_epoch_test
                        checkpoint_dict['best_gen_test'] = self.best_gen_test
                        checkpoint_dict['cur_test_acc'] = cur_test_acc1

                        test_summary = {
                            'Test/Loss':cur_test_loss,
                            'Test/Acc1':cur_test_acc1,
                            'Test/Acc5':cur_test_acc5,
                            'Best/Best Test Acc1': self.best_test_acc1,
                            'Test/gen':gen,
                            'Best/gen':gen,
                            'Test/epoch_in_gen': epoch,
                            'Test/epoch': epoch + (gen*self.cfg.epochs)
                            }
                        summary_logging(self.cfg, test_summary, Stage.TEST)

                        if is_best_test_gen and not self.cfg.debug:
                            save_checkpoint(self.cfg, checkpoint_dict, file_name='best_test.pt')
                    
                    # On epoch end
                    # Checkpointing
                    if not self.cfg.debug:
                        save_checkpoint(self.cfg, checkpoint_dict, file_name='model.pt')
                        if epoch == self.ascent_over_epoch:
                            save_checkpoint(self.cfg, checkpoint_dict, file_name='after_ascent.pt')
                        if epoch == self.ascent_over_epoch + self.cfg.save_after_k_epochs_ascentOver:
                            k = self.cfg.save_after_k_epochs_ascentOver
                            save_checkpoint(self.cfg, checkpoint_dict, file_name=f'{k}_epochs_after_ascent.pt')
                            
                    if is_best_val_gen and not self.cfg.debug:
                        save_checkpoint(self.cfg, checkpoint_dict, file_name='best_val.pt')

                

                    
            # On GEN end
            print('-'*70)
            print(f'Best Val acc in gen {gen} was {self.gen_best_val_acc:6.4f} at epoch {epoch:03d} and overall best val acc1 is {self.best_val_acc1:6.4f}')
            if test_loader is not None:
                print(f'Best Test acc in gen {gen} was {self.gen_best_test_acc:6.4f} at epoch {epoch:03d} and overall best val acc1 is {self.best_test_acc1:6.4f}')
            print('-'*70)

        # On training end
        print('-'*70)
        print(f'Best overall Val acc was {self.best_val_acc1:6.4f} at gen {self.best_gen_val} at epoch {self.best_epoch_val:03d}')
        if test_loader is not None:
            print(f'Best overall Test acc was {self.best_test_acc1:6.4f} at gen {self.best_gen_test} at epoch {self.best_epoch_test:03d}')
        print('-'*70)





    def grad_norms_logging(self, iter_idx):
        if self.track_gradient_list is None: # Nothing to track
            return
        _grad_track_dict = { 
            '_'.join(k.split('.')[:3] + ['grad']): torch.norm(rgetattr(self.model, k).grad, p=2) #/ torch.prod(torch.tensor(rgetattr(self.model, k).shape), 0) 
            for k in self.track_gradient_list
            }

        
        # At start of each epoch reinitialize it
        if iter_idx == 0:        
            _norm_track_dict = { 
                '_'.join(k.split('.')[:3] + ['norm']): torch.norm(rgetattr(self.model, k), p=2) #/ torch.prod(torch.tensor(rgetattr(self.model, k).shape), 0) 
                for k in self.track_gradient_list
            }
            self._first_grad_dict = False
            self.grad_track_dict = {
                k:v.cpu().item() for k,v in _grad_track_dict.items()
            }
            self.norm_track_dict = {
                k:v.cpu().item() for k,v in _norm_track_dict.items()
            }
        else:
            self.grad_track_dict = {
                k: v.cpu().item() + self.grad_track_dict[k] for k,v in _grad_track_dict.items()
            }
        
    def _flush_grad_norms(self, epoch):
        if self.track_gradient_list is None:
            return
        if hasattr(self, 'grad_track_dict'):
            grad_dict = dict(self.grad_track_dict)
            grad_dict['epoch'] = epoch 
            wandb_logging(self.cfg, grad_dict)
        if hasattr(self, 'norm_track_dict'):
            norm_dict = dict(self.norm_track_dict)
            norm_dict['epoch'] = epoch 
            wandb_logging(self.cfg, norm_dict)
        
    def _get_lr_scheduler(self, optimizer):
        cfg = self.cfg
        _lr_policy = get_policy(cfg.lr_policy)
        lr_policy = _lr_policy(
            optimizer=self.optimizer,
            warmup_length=cfg.warmup_length,
            lr=cfg.lr,
            epochs=cfg.epochs, # Following parameters are possibly needed
            num_generations=cfg.num_generations,
            current_gen=self.start_gen,
        )
        return lr_policy
