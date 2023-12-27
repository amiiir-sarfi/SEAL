import argparse
from pathlib import Path
from utils.utils import get_checkpoints_dir

args = None


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Iterative Training")
        # Data parameters
        parser.add_argument(
            "--data_dir", 
            default='.../ssd/tiny_imagenet/tinyImagenet_full/', 
            help="path to dataset ROOT directory"
        )        
        parser.add_argument( 
            "--set",
            type=str,
            default="tinyImagenet_full",
            help="Name of the dataset"
        )
        parser.add_argument(
            "--num_cls",
            default=200,
            type=int,
            help="Number of workers for dataloader",
        )
        parser.add_argument(
            "--eval_tst",
            default=0,
            type=int,
            help="If there are three sets for this dataset (train/val/test), this flag should be 1"
            "If there are 2 sets for this dataset (train/test), it should be false. "
            "If neither (e.g., you only have train set for your dataset), the code will not run"
            "A quick fix would be to generate a very small dummy test set (e.g., torch.rand(10,3,64,64)",
        )
        parser.add_argument(
            "--num_workers",
            default=6,
            type=int,
            help="Number of workers for dataloader",
        )
        parser.add_argument(
            "--all_in_ram",
            default=0,
            type=int,
            help="Load the whole dataset in ram (TRUE) or just the addresses (FALSE)?",
        )

        # Training parameters
        parser.add_argument( 
            "-a",
            "--arch",
            metavar="ARCH",
            default="ResNet50",
            choices=["ResNet18", "ResNet50"],
        )
        parser.add_argument( #
            "--num_generations",
            default=10,
            type=int,
            help="Number of training generations",
        ) 
        parser.add_argument( 
            "--epochs", default=160, type=int, help="number of total epochs per gen"
        )
        parser.add_argument( 
            "-b",
            "--batch_size",
            default=32,
            type=int,
            metavar="N",
            help="mini-batch size",
        )
        parser.add_argument(
            "--val_interval",
            default=2,
            type=int,
            help="Eval on val split every ? epochs",
        )        
        parser.add_argument( #
            "--exp_mode",
            type=str,
            default="ascend",
            choices=["ascend", "fortuitous", "fortuitous+ascend", "normal"],
            help="What method to use? ascend refers to SEAL and fortuitous refers to LLF"
            "fortuitous+ascend was used for the ablation studies",
        )

        # main params
        parser.add_argument(
            "--layer_threshold", #
            default="layer3",
            type=str,
            help="layer to start resetting (for SEAL and LLF). "
            "E.g., layer3 means blocks 3 and 4 should be reinitialized in ResNet in LLF"
            "and it means blocks 1 and 2 should do the intermittent gradient ascent",
        )        
        parser.add_argument( #
            "--ascending_epochs",
            default=40,
            type=float,
            help="how many epochs should we ascend? (usually 1/4 of max_epochs works well)",
        )
        parser.add_argument( #
            "--ascenders_lr_multiplier",
            default= -0.01,
            type=float,
            help="Variable S in the paper (the negative sign is used to indicate gradient ascent",
        ) 
        parser.add_argument( #
            "--ascenders_wd_multiplier",
            default=0,
            type=float,
            help="During ascending, what should happen to weight decay? Positive values fail (refer to paper in proposed method)",
        )

        # For ablation studies
        parser.add_argument( #
            "--reverse",
            default=0,
            type=int,
            help="Reverse the fit and forgetting hypotheses? (used in an ablation study)",
        )
        parser.add_argument( #
            "--fc_never_ascend",
            default=0,
            type=int,
            help="Should FC ever do ascend? (for the reverse case)"
            "We tried both False and True for this flag and neither worked!",
        )
        parser.add_argument( #
            "--freeze_descenders",
            default=0,
            type=int,
            help="While ascending, should the rest of the layers descend or freeze? (used in an ablation study)",
        )

        parser.add_argument(
            "--save_after_k_epochs_ascentOver",
            default=200,
            type=float,
            help="Save a few epochs after ascent is over (for analysis).",
        )

        parser.add_argument(
            "--auto_mix_prec",
            default=0,
            type=int,
            help="This flag enables training with automatic mixed-precision. ",
        )

        # Optimizer parameters
        parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
        parser.add_argument("--lr", default=0.1, type=float, metavar="LR", dest="lr") #

        parser.add_argument("--weight_decay", default=5e-4, type=float, metavar="W") #
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M") #

        # LR Scheduler parameters
        parser.add_argument( 
            "--lr_policy", default="cosine_lr", help="Policy for the learning rate."
        )

        parser.add_argument( 
            "--warmup_length",
            default=20,
            type=int,
            help="Number of warmup iterations",
        )
        # Loss parameters
        parser.add_argument(
            "--criterion",
            default="CE",
            type=str,
            choices=["CE", "cs_kd"],
            help="which criterion to use? We always used CE+label-smoothing (following LLF on Tiny-ImageNet)",
        )
        parser.add_argument( #
            "--label_smoothing",
            type=float,
            help="Label smoothing to use, default 0.1"
            "CAREFUL! if cs_kd is on, label smoothing would be disabled in parser.parse",
            default=0.1,
        )

        # Checkpointing
        parser.add_argument(
            "--ckpt_dir",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint",
        )
        parser.add_argument(
            "--resume_from_next_gen",
            default=1,
            type=int,
            help="Whether to resume from current gen/epoch or next gen with epoch=0 (when loading from checkpoint)",
        )

        parser.add_argument(
            "--exp_dir",
            default="./runs/tiny_exp",
            help="Directory to save checkpoints and summaries",
        )

        # wandb
        parser.add_argument(
            "--no_wandb", type=int, default=0, help="no wandb"
        )
        parser.add_argument(
            "--wandb_project_name",
            type=str,
            default="TinyImageNet_test",
            help="Wandb sweep may overwrite this!",
        )
        parser.add_argument("--wandb_experiment_name", type=str, default="")
        parser.add_argument("--wandb_entity", type=str)
        parser.add_argument("--wandb_offline", type=str, default="offline")

        # debugging
        parser.add_argument(
            "--debug",
            default=0,
            type=int,
            help="Run the experiment with only a few batches for all"
            "datasets, to ensure code runs without crashing.",
        )
        parser.add_argument(
            "--debug_batches",
            type=int,
            default=2,
            help="Number of batches to run in debug mode.",
        )
        parser.add_argument(
            "--debug_epochs",
            type=int,
            default=35,
            help="Number of epochs to run in debug mode. "
            "If a non-positive number is passed, all epochs are run.",
        )
        parser.add_argument(
            "--debug_gens",
            type=int,
            default=3,
            help="Number of gens to run in debug mode. "
            "If a non-positive number is passed, all epochs are run.",
        )
        
        self.parser = parser

    def parse(self, args):
        self.cfg = self.parser.parse_args(args)

        self.cfg.data_dir = Path(self.cfg.data_dir)


        if self.cfg.exp_mode == 'normal':
            self.cfg.ascenders_lr_multiplier = 1
            self.cfg.ascenders_wd_multiplier = 1
            current_ascending_epochs = self.cfg.ascending_epochs
            self.cfg.save_after_k_epochs_ascentOver += current_ascending_epochs
            self.cfg.ascending_epochs = 0

        if self.cfg.exp_mode == 'fortuitous':
            self.cfg.ascenders_lr_multiplier = 1
            self.cfg.ascenders_wd_multiplier = 1

        # For CS_KD. Adapted from LLF codebase
        if self.cfg.criterion == "cs_kd":
            self.cfg.samples_per_class = 2
            self.cfg.label_smoothing = 0

        if self.cfg.debug:
            self.cfg.val_interval = 2

        # directory to save ckpts in
        self.cfg.checkpoints_dir = get_checkpoints_dir(
            self.cfg.exp_dir, self.cfg.set, self.cfg.ascenders_lr_multiplier, self.cfg.ascenders_wd_multiplier, self.cfg.exp_mode
        )

        # file to load ckpt from
        if self.cfg.ckpt_dir:
            self.cfg.ckpt_dir = Path(self.cfg.ckpt_dir)

        print(self.cfg)
        return self.cfg

if __name__ == "__main__":
    config = Config().parse(None)
