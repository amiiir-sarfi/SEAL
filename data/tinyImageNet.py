from pathlib import Path
from PIL import Image
from typing import Any, Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms as T


class TinyImageNet(Dataset):
    def __init__(
        self,
        data_path: str = '$DATASETS/tiny_imagenet/tinyImagenet_full',
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 64,
        all_in_ram: bool = False
    ):
        """Create a TinyImageNet dataset from a directory with congruent text and image names.
        Args:
            images_path (str): Path to the folder containing images of the dataset.
            ann_file_path (str): Path to the `json` or `csv` annotation file.
            transform (_type_, optional): _description_. Defaults to None.
            image_size (Optional[int], optional): The size of outputted images.. Defaults to 64.
            resize_ratio (Optional[float], optional): Minimum percentage of image contained by resize. Defaults to 0.75.
        """
        super(TinyImageNet, self).__init__()

        split = split.lower()
        assert split in ['train', 'val'], f"split should be train/val but given {split}"
        
        cls_name_to_idx = {}
        cls_idx_to_name = {}
        with open(Path(data_path, 'wnids.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                cls_name_to_idx[line.strip()] = idx
                cls_idx_to_name[idx] = line.strip()
        
        data_path = Path(data_path) / split
        self.dataset = []
        if split == 'train':
            train_paths = [
                *data_path.glob("**/*.JPEG"),
                *data_path.glob("**/*.png"),
                *data_path.glob("**/*.jpg"),
                *data_path.glob("**/*.jpeg"),
                *data_path.glob("**/*.bmp")
            ]
            train_paths = sorted(train_paths)
            for img_path in train_paths:
                label = img_path.parts[-3]
                if all_in_ram:
                    img_path = Image.open(img_path).convert("RGB")
                self.dataset.append((img_path, cls_name_to_idx[label]))
                
        else:
            with open(data_path / 'val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    file, label = line.split()[0:2]
                    img_path = data_path / 'images' / file
                    if all_in_ram:
                        img_path = Image.open(img_path).convert("RGB")
                    self.dataset.append((img_path, cls_name_to_idx[label]))
        

        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.targets = [v for _,v in self.dataset]
        self.all_in_ram = all_in_ram
        self.cls_name_to_idx = cls_name_to_idx
        self.cls_idx_to_name = cls_idx_to_name
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not self.all_in_ram:
            img = Image.open(img).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, label
            
            
    def __len__(self):
        return len(self.dataset)    
    

if __name__ == "__main__":
    img, label = TinyImageNet(
        data_path='/ssd/tiny_imagenet/tinyImagenet_full', 
        split='val'
        )[2]
    
    print(img[0].shape, label)
