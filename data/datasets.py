import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as T
import torchvision
from collections import defaultdict
import random
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from PIL import ImageEnhance

from .tinyImageNet import TinyImageNet

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices
#             yield list(itertools.chain(*zip(batch_indices,pair_indices )))

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class ImageJitter(object):
    def __init__(self, transformdict=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def _get_mean_std(cfg):
    if cfg.set.lower() == 'cifar10':
        mean_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif cfg.set.lower() == 'cifar100':
        mean_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif 'mnist' in cfg.set.lower():
        mean_std = (0.1307), (0.3081)
    elif cfg.set.lower() == 'imagenet64':
        mean_std = (0.482, 0.458, 0.408), (0.269, 0.261, 0.276)
    elif cfg.set.lower() == 'imagenet32':
        mean_std = (0.481, 0.457, 0.408), (0.260, 0.253, 0.268)
    else: # e.g., TINY IMAGENET:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
    return mean_std


def get_transforms(cfg):
    if cfg.set == 'tinyImagenet_full':
        transform_train = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
        ])
    else:
        raise NotImplementedError(f"No transformations exists for {cfg.set} dataset")
    return transform_train, transform_test


class DatasetWrapper(torch.utils.data.Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices
            
        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, torchvision.datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels
                    
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1        

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def load_dataset(cfg):
    transform_train, transform_test = get_transforms(cfg)
    testset = None
    testloader = None
    if cfg.set=='tinyImagenet_full':
        trainset = TinyImageNet(cfg.data_dir, split='train', transform=transform_train, all_in_ram=cfg.all_in_ram)
        valset = TinyImageNet(cfg.data_dir, split='val', transform=transform_test, all_in_ram=cfg.all_in_ram)
    
    if cfg.criterion == "cs_kd":
        get_train_sampler = lambda d: PairBatchSampler(d, cfg.batch_size)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), cfg.batch_size, False)
    else:
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), cfg.batch_size, False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), cfg.batch_size, False)

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=cfg.num_workers, pin_memory=True)
    valloader = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=cfg.num_workers, pin_memory=True)
    
    if testset is not None:
        testloader = DataLoader(testset,   batch_sampler=get_test_sampler(testset), num_workers=cfg.num_workers, pin_memory=True)
    
    return trainloader, valloader, testloader
    
    
    
if __name__ == '__main__':
    from config import Config
    config = Config().parse(None)
    config.batch_size = 32
    print(config)
    print(config.set)
    print(config.data_dir)
    tl, vl, _ = load_dataset(config)
    print(len(tl))
    batch = next(iter(tl))