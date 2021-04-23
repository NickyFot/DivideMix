import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch


class AffectNet(Dataset):
    def __init__(
            self,
            root_dir: str,
            img_transform: transforms,
            target_transform: transforms,
            annotation_filename: str,
            mode: str = None,
            pred: list = [],
            probability: list = [],
            **kwargs
    ):
        self.root = root_dir+'Manually_annotated/extracted_cropped'
        self.annotation_path = os.path.join(*[root_dir, 'Manually_Annotated', annotation_filename])
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
        img_pth = self.data['images'][idx]['file_name']
        img = Image.open(self.root + img_pth).convert('RGB')
        img = self.transform(img)
        target = self.data['annotations'][idx]
        target = self.target_transform(target)
        return img, target, idx

    def _getitem_unlabeled(self, idx):
        img_pth = self.data['images'][idx]['file_name']
        img = Image.open(self.root + img_pth).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def _getitem_labeled(self, idx):
        img_pth = self.data['images'][idx]['file_name']
        img = Image.open(self.root + img_pth).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        target = self.data['annotations'][idx]
        target = self.target_transform(target)
        prob = self.probability[idx]
        return img1, img2, target, prob

    def _clean_data(self, filter_expressions: list = None, partition: str = None):
        if filter_expressions:
            new_img = list()
            new_annot = list()
            for idx, datum in self.data['annotations']:
                expression = datum['expression']
                if expression in filter_expressions:
                    new_img.append(self.data['images'][idx])
                    new_annot.append(self.data['annotations'][idx])
                self.data['images'] = new_img
                self.data['annotations'] = new_annot
        if partition:
            clean = list()
            noisy = list()
            for idx, datum in self.data['annotations']:
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


class AffectNetDataloader(object):
    def __init__(self, batch_size, num_workers, root_dir, log):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log

        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.transform_val = transforms.Compose([
                transforms.Resize(320),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.target_transform = transforms.Compose([torch.FloatTensor])
        self.filter_expression = list(range(8))

    def run(self, mode: str, pred: list = [], prob: list = []):
        if mode == 'warmup':
            all_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_regression_ext_det.json',
                target_transform=self.target_transform,
                filter_expressions=self.filter_expression
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return trainloader

        elif mode == 'train':
            labeled_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_regression_ext_det.json',
                target_transform=self.target_transform,
                mode='labeled',
                pred=pred,
                probability=prob,
                filter_expressions=self.filter_expression
            )
            unlabeled_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_regression_ext_det.json',
                target_transform=self.target_transform,
                mode='unlabeled',
                pred=pred,
                probability=prob,
                filter_expressions=self.filter_expression
            )
            labeledloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            unlabeledloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return labeledloader, unlabeledloader
        elif mode == 'test':
            test_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_val_all_regression_ext_det.json',
                target_transform=self.target_transform,
                filter_expressions=self.filter_expression
            )
            testloader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return testloader
        elif mode == 'eval_train':
            eval_dataset = AffectNet(
                self.root_dir,
                img_transform=self.transform_train,
                annotation_filename='affectnet_annotations_train_all_regression_ext_det.json',
                target_transform=self.target_transform,
                filter_expressions=self.filter_expression
            )
            evalloader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return evalloader
