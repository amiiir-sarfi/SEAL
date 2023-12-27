import wandb

from config import Config
from pathlib import Path
from core import IterativeTrainer

import warnings
warnings.filterwarnings("ignore")

import torch

if __name__ == "__main__":
    print(f'cuda is {torch.cuda.is_available()}')
    assert torch.cuda.is_available(), "Cuda must be available"
    
    cfg = Config().parse(None)
    cfg.file_name = Path(__file__).name
    
    # exp_mode in ['reset', 'ascend'(ours), 'random_ascend']
    cfg.run_info = f"exp{cfg.exp_mode}_{cfg.criterion}_reverse{cfg.reverse}"
    
    if not cfg.no_wandb:
        wandb.init(
            project=cfg.wandb_project_name, entity=cfg.wandb_entity,
            name=cfg.wandb_experiment_name, config=cfg, mode = cfg.wandb_offline
            )
        wandb.config.update(cfg)
    else:
        wandb = None
    
    trainer = IterativeTrainer(cfg)
        
    trainer.fit(progressbar=False)