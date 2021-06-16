import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import cv2
import torch


class Digitize(object):
    def __init__(self, range: tuple, step: float):
        self._start = range[0]
        self._end = range[1]
        self._boundaries = torch.arange(self._start, self._end, step)

    def __call__(self, x: torch.tensor) -> torch.tensor:
        return torch.bucketize(x, boundaries=self._boundaries) - 1


class ReplaceValues(object):
    def __init__(self, value, new_value):
        self.val = value
        self.new_val = new_value

    def __call__(self, x):
        shape = x.size()
        random_x = torch.rand(shape)
        random_x = -2 * random_x + 1
        mask = x == self.val
        new_x = x.clone()
        if self.new_val:
            new_x[mask] = self.new_val
        else:
            new_x[mask] = random_x[mask]
        return new_x


class ColumnSelect(object):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, feats: dict):
        feats = np.array([feats.get(key) for key in self.keys])
        return feats


class AffectNet(Dataset):
    def __init__(
            self,
            root_dir: str,
            img_transform: transforms,
            target_transform: transforms,
            annotation_filename: str,
            mode: str = None,
            pred: list = None,
            probability: list = None,
            **kwargs
    ):
        self.root = root_dir
        self.annotation_path = annotation_filename
        self.transform = img_transform
        self.target_transform = target_transform
        self.mode = mode
        self.data = json.load(open(self.annotation_path))
        self._clean_data(**kwargs)
        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
            self.data['images'] = [x for idx, x in enumerate(self.data['images']) if idx in pred_idx]
            self.data['annotations'] = [x for idx, x in enumerate(self.data['annotations']) if idx in pred_idx]
        elif self.mode == 'unlabeled':
            pred_idx = (1 - pred).nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
            self.data['images'] = [x for idx, x in enumerate(self.data['images']) if idx in pred_idx]
            self.data['annotations'] = [x for idx, x in enumerate(self.data['annotations']) if idx in pred_idx]

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        if self.mode == 'labeled':
            return self._getitem_labeled(idx)
        elif self.mode == 'unlabeled':
            return self._getitem_unlabeled(idx)
        else:
            return self._getitem_base(idx)

    def _getitem_base(self, idx):
        img_pth = os.path.join(*[self.root, 'Manually_Annotated', 'extracted_cropped', self.data['images'][idx]['file_name']])
        img = cv2.imread(img_pth)
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)
        target = self.data['annotations'][idx]
        target = self.target_transform(target)
        return img, target, idx

    def _getitem_unlabeled(self, idx):
        img_pth = os.path.join(*[self.root, 'Manually_Annotated', 'extracted_cropped', self.data['images'][idx]['file_name']])
        img = cv2.imread(img_pth)
        img = Image.fromarray(img).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def _getitem_labeled(self, idx):
        img_pth = os.path.join(*[self.root, 'Manually_Annotated', 'extracted_cropped', self.data['images'][idx]['file_name']])
        img = cv2.imread(img_pth)
        img = Image.fromarray(img).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        target = self.data['annotations'][idx]
        target = self.target_transform(target)
        prob = self.probability[idx]
        return img1, img2, target, prob

    def _clean_data(self, filter_expressions: list = None, partition: str = None, artifitial_noise: str = None):
        if filter_expressions:
            new_img = list()
            new_annot = list()
            for idx, datum in enumerate(self.data['annotations']):
                expression = datum['expression']
                if expression in filter_expressions:
                    new_img.append(self.data['images'][idx])
                    new_annot.append(self.data['annotations'][idx])
            self.data['images'] = new_img
            self.data['annotations'] = new_annot
        if partition:
            clean = list()
            noisy = list()
            for idx, datum in enumerate(self.data['annotations']):
                expression = datum['expression']
                arousal = datum['arousal']
                valence = datum['valence']
                intensity = np.sqrt(valence ** 2 + arousal ** 2)
                if expression == 0 and intensity >= 0.2:
                    noisy.append(idx)
                elif expression == 1 and (valence <= 0 or intensity <= 0.2):
                    noisy.append(idx)
                elif expression == 2 and (valence >= 0 or intensity <= 0.2):
                    noisy.append(idx)
                elif expression == 3 and (arousal <= 0 or intensity <= 0.2):
                    noisy.append(idx)
                elif expression == 4 and (not (arousal >= 0 and valence <= 0) or intensity <= 0.2):
                    noisy.append(idx)
                elif expression == 5 and (valence >= 0 or intensity <= 0.3):
                    noisy.append(idx)
                elif expression == 6 and (arousal <= 0 or intensity <= 0.2):
                    noisy.append(idx)
                elif expression == 7 and (valence >= 0 or intensity <= 0.2):
                    noisy.append(idx)
                else:
                    clean.append(idx)
            if partition == 'clean':
                self.data['images'] = [img for idx, img in enumerate(self.data['images']) if idx in clean]
                self.data['annotations'] = [img for idx, img in enumerate(self.data['annotations']) if idx in clean]
            else:
                self.data['images'] = [img for idx, img in enumerate(self.data['images']) if idx in noisy]
                self.data['annotations'] = [img for idx, img in enumerate(self.data['annotations']) if idx in noisy]
        if artifitial_noise:
            noisy = json.load(open('noisy_idx.json'))
            clean = [idx for idx, datum in enumerate(self.data['annotations']) if datum['id'] not in noisy]


class AffectNetDataloader(object):
    def __init__(self, batch_size, num_workers, root_dir, log, **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.kwargs = kwargs

        self.transform_train = transforms.Compose([
                transforms.Resize(98),
                transforms.RandomResizedCrop(92),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAutocontrast(p=.25),
                transforms.RandomRotation(degrees=25),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.transform_val = transforms.Compose([
                transforms.Resize(98),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.target_transform = transforms.Compose([
            ColumnSelect(['arousal', 'valence', 'expression']),
            torch.FloatTensor,
            # ReplaceValues(-2, None),
            # Digitize(range=(-1, 1.01), step=0.1),
            # torch.LongTensor
        ])
        self.filter_expression = list(range(8))
        # self.filter_expression.append(9)  # train on uncertain

        self.filter_expression_test = list(range(8))

    def run(self, mode: str, pred: list = [], prob: list = []):
        if mode == 'warmup':
            all_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_ext_det.json',
                target_transform=self.target_transform,
                mode=None,
                filter_expressions=self.filter_expression,
                **self.kwargs
            )
            # debug line
            print('# Train Images ' + str(len(all_dataset)))
            train_loader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return train_loader

        elif mode == 'train':
            labeled_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_ext_det.json',
                target_transform=self.target_transform,
                mode='labeled',
                pred=pred,
                probability=prob,
                filter_expressions=self.filter_expression,
                **self.kwargs
            )
            unlabeled_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_ext_det.json',
                target_transform=self.target_transform,
                mode='unlabeled',
                pred=pred,
                probability=prob,
                filter_expressions=self.filter_expression,
                **self.kwargs
            )
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return labeled_loader, unlabeled_loader
        elif mode == 'test':
            test_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_val_all_ext_det.json',
                mode=None,
                target_transform=self.target_transform,
                filter_expressions=self.filter_expression_test,
                **self.kwargs
            )
            # debug line
            print('# Test Images: ' + str(len(test_dataset)))
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return test_loader
        elif mode == 'eval_train':
            eval_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_ext_det.json',
                mode=None,
                target_transform=self.target_transform,
                filter_expressions=self.filter_expression,
                **self.kwargs
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return eval_loader
