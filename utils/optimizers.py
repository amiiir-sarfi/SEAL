import torch

def get_optimizer(optim_type, param_groups, lr=0.1, momentum=0.9, weight_decay=0):
    # for n, v in model.named_parameters():
    #     if v.requires_grad:
    #         args.logger.info("<DEBUG> gradient to {}".format(n))

    #     if not v.requires_grad:
    #         args.logger.info("<DEBUG> no gradient to {}".format(n))

    if optim_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr,
                                          momentum=momentum, weight_decay=weight_decay)
        
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, param_groups), lr=lr
        )
    elif optim_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=lr, alpha=0.9, weight_decay = weight_decay, momentum = 0.9)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay = weight_decay)
    else:
        raise NotImplemented('Invalid Optimizer {}'.format(optim_type))

    return optimizer

def get_optimizer_fine_tune(args, model,fine_tune=False,criterion=None):
    # for n, v in model.named_parameters():
    #     if v.requires_grad:
    #         args.logger.info("<DEBUG> gradient to {}".format(n))

    #     if not v.requires_grad:
    #         args.logger.info("<DEBUG> no gradient to {}".format(n))

    param_groups = model.parameters()
    if fine_tune:
        # Train Parameters
        param_groups = [
            {'params': list(
                set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu != -1 else
            list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
            {
                'params': model.model.embedding.parameters() if args.gpu != -1 else model.module.model.embedding.parameters(),
                'lr': float(args.lr) * 1},
        ]
        if args.ml_loss == 'Proxy_Anchor':
            param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optimizer == "sgd_TEMP": #use this for freeze layer experiments, so there are no parameter updates for frozen layers
        parameters = list(model.named_parameters())
        param_groups = [v for n, v in parameters if v.requires_grad]        
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)     
        
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, param_groups), lr=args.lr
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay = args.weight_decay)
    else:
        raise NotImplemented('Invalid Optimizer {}'.format(args.optimizer))

    return optimizer