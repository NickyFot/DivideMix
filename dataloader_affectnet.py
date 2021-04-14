import os

import cv2

import numpy as np

import albumentations as A

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import sys
sys.dont_write_bytecode = True

class affectnet_dataloader():
    def __init__(self, args, cfg, phase):
        self.args = args
        self.cfg = cfg
        self.phase = phase

    def run(self, mode=None, pred=[], prob=[]):
        return init_dataset(self.args, self.cfg, phase=self.phase)



def init_dataset(args, cfg, phase='train'):
    dataset = AffectNet(args, cfg, phase, batch_size=args.batch_size*2)

    return dataset


class AffectNet():

    def __init__(self, args, cfg, phase, batch_size):
        self.batch_size = batch_size
        self.phase = phase

        try:
            self.rank = args.rank
        except:
            self.rank = False

        self.balance = args.balance
        self.regression_head_ids = cfg['regression_head_ids']
        self.input_size = cfg['input_size']

        ###################################
        if self.phase == 'train':
            self.transform = A.Compose([
                A.Rotate(limit=25, always_apply=False, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomScale(scale_limit=0.15, always_apply=False, p=0.5),
                # A.Resize(256, 256),
                # A.RandomCrop(width=240, height=240, always_apply=False, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(always_apply=False, p=0.25),
                # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=0.25)#,
                A.Resize(self.input_size, self.input_size)
            ])
        ###################################

        if args.remote:
            self.root_affectnet = cfg['root_affectnet_remote']
        else:
            self.root_affectnet = cfg['root_affectnet_local']
        self.dir_images = os.path.join(self.root_affectnet, 'Manually_Annotated', 'extracted_cropped')
        root_metadata = os.path.join(os.getcwd(), 'metadata')

        if self.phase == 'train':
            npy_training = os.path.join(root_metadata, 'training_8_classes.npy')
            self.metadata = np.load(npy_training)
        elif self.phase == 'val':
            # npy_validation = os.path.join(root_metadata, 'validation_8_classes.npy')
            npy_validation = os.path.join(root_metadata, 'validation_4000.npy')
            self.metadata = np.load(npy_validation)
        self.N_samples = len(self.metadata)

        self.reset_indices()
        self.create_pools()

    def __iter__(self):
        return self.get_batch(self.batch_size)

    def __len__(self):
        return self.N_samples

    def augment_new(self, image):

        # image = image[:, :, [2,1,0]]

        # plt.imshow(image)
        # plt.show()

        transformed = self.transform(image=image)
        transformed_image = transformed["image"]

        # plt.imshow(transformed_image)
        # plt.show()

        return transformed_image

    def reset_indices(self):

        # print('\nResetting dataset indices...\n')
        self.indices = list(np.random.permutation(self.N_samples))

    def create_pools(self):
        self.expressions = []
        for el in self.metadata:
            self.expressions.append(int(el[1]))
        self.expressions = np.array(self.expressions)
        self.unique_classes = np.unique(self.expressions)
        self.N_exp = len(self.unique_classes)

        sum_n_samples = 0
        self.pools = {}
        self.n_pool_samples = {}
        for exp in self.unique_classes:
            exp_str = str(exp)
            inds_exp = np.where(self.expressions == exp)[0]
            # print(len(inds_exp))
            sum_n_samples += len(inds_exp)
            self.pools[exp_str] = inds_exp
            self.n_pool_samples[exp_str] = len(inds_exp)

    # print(sum_n_samples)

    def get_pool_sample(self):

        exp = np.random.choice(self.N_exp)
        exp_str = str(exp)
        n_samples = self.n_pool_samples[exp_str]
        pool_index = np.random.choice(n_samples)
        index = self.pools[exp_str][pool_index]

        return exp

    def augment(self, image):

        if self.input_size == 224:
            # resize to 230x230 --> 224
            image = cv2.resize(image, (230, 230), interpolation=cv2.INTER_AREA)
        else:
            # resize to 120x120 --> 112x112
            image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)

        # crop sizes
        w = image.shape[1]
        h = image.shape[0]
        cut_w = np.random.randint(low=0, high=6)
        cut_h = np.random.randint(low=0, high=6)
        # do cropping
        image = image[cut_h:h - cut_h, cut_w:w - cut_w, :]

        # flip number
        flip_number = np.random.randint(low=0, high=2)
        # do horizontal flipping
        if flip_number == 1:
            image = image[:, ::-1, :]

        return image

    def get_batch(self, batch_size):

        epoch_end = False

        images = []
        labels_regr = []
        labels_cls = []

        for i in range(batch_size):

            #################################
            if self.phase == 'train' and self.balance:
                index = self.get_pool_sample()
            else:
                index = self.indices[-1]
                self.indices.pop(-1)
            #################################

            info = self.metadata[index]
            img_path = info[0]
            img_path_full = os.path.join(self.dir_images, img_path)
            image = cv2.imread(img_path_full)
            if self.phase == 'train':
                image = self.augment_new(image)
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

            label = info[1:]
            exp = int(label[0])
            arousal = float(label[1])
            valence = float(label[2])
            # label = [exp, arousal, valence]
            label_regr = []
            for id_string in self.regression_head_ids:
                if id_string == 'arousal':
                    label_regr.append(arousal)
                elif id_string == 'valence':
                    label_regr.append(valence)
            images.append(image)
            labels_regr.append(label_regr)
            labels_cls.append(exp)

        images = np.array(images).astype(np.float32)
        labels_regr = np.array(labels_regr)
        labels_cls = np.array(labels_cls)

        images /= 255.0
        images = np.transpose(images, (0, 3, 1, 2))

        if len(self.indices) < batch_size:
            self.reset_indices()
            epoch_end = True

        return images, labels_regr, labels_cls, epoch_end


def init_args_cfg(args):
    cfg = cfg_affectnet
    if args.regression:
        # dimensional model, regression
        cfg['num_classes_regr'] = 2
    if args.var:
        cfg['num_classes_var'] = 2
    cfg['input_size'] = 112
    return args, cfg

cfg_affectnet = {
	'name': 'affectnet',
	'root_affectnet_local': '/home/archniki/Datasets/AffectNet',
	'root_affectnet_remote': '/home/lab/datasets/AffectNet',
	'path_manual': 'Manually_Annotated',
	'path_auto': 'Automatically_Annotated',
	# 'keypoints': 'face',
	'emotion_classes': ['Neutral',
						'Happy',
						'Sad',
						'Surprise',
						'Fear',
						'Disgust',
						'Anger',
						'Contempt',
						'None',
						'Uncertain',
						'Non-Face'],
	'means': (0.0, 0.0, 0.0),
	'stds': (255.0, 255.0, 255.0),
	'input_size': 100,
	'dims_hidden': 256,
	'dims_rank': 3,
	'dims_feats': 256, # 512*3*3,
	'regression_head_ids': ('arousal', 'valence'),
	'target_names': ('arousal', 'valence'),
	'basenet': 'vgg16',
	'max_epoch': 200,
}