# Simulated annealing in EArly Layers Leads to Better Generalization (SEAL)

This repo is associated with the paper [Simulated Annealing in Early Layers Leads to Better Generalization](https://arxiv.org/abs/2304.04858) (CVPR 2023).

### Install dependencies with pip
```bash
python3 -m venv venv
source venv/bin/activate

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Getting Started

```
python main.py --set tinyImagenet_full --data_dir=<u>DataRoot</u> --arch=ResNet50 --exp_mode=ascend 
```

Refer to [config.py](./config.py) if you wish to try different hyper parameters/settings. 

