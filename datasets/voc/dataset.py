# copy and modified from https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py

import os.path as osp
import os

import numpy as np
import PIL.Image
import scipy.io

from torch.utils import data
import datasets.joint_transforms as jttransforms
import torchvision.transforms.transforms as tvtransforms


class VOCClassSegBase(data.Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    default_joint_transforms_train = jttransforms.Compose([jttransforms.Scale(600),
                                                           jttransforms.RandomCrop(512),
                                                           jttransforms.RandomHorizontallyFlip()])
    default_transforms = tvtransforms.Compose([tvtransforms.ToTensor(), tvtransforms.Normalize(mean=mean, std=std)])
    default_target_transforms = lambda lbl: np.array(lbl, dtype=np.int32).astype(np.int64)

    def __init__(self, root, joint_transforms=default_joint_transforms_train, transforms=default_transforms,
                 target_transforms=default_target_transforms, split='train'):
        self.root = root
        self.joint_transforms = joint_transforms
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.imgs = list()
        self._make_dataset(split)

    def _make_dataset(self, split):
        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        imgsets_file = osp.join(
            dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(
                dataset_dir, 'SegmentationClass/%s.png' % did)
            self.imgs.append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data_file = self.imgs[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)

        lbl = self.loadlbl(data_file['lbl'])

        if self.joint_transforms:
            img, lbl = self.joint_transforms(img, lbl)

        if self.transforms:
            img = self.transforms(img)

        if self.target_transforms:
            lbl = self.target_transforms(lbl)

        return img, lbl

    def loadlbl(self, lbl_file):
        # load label
        return PIL.Image.open(lbl_file)



class VOC2011ClassSeg(VOCClassSegBase):
    def __init__(self, root, split='train'):
        super(VOC2011ClassSeg, self).__init__(root, split=split)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.imgs['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train'):
        super(VOC2012ClassSeg, self).__init__(root, split=split)


class SBDClassSeg(VOCClassSegBase):
    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train'):
        super(SBDClassSeg, self).__init__(root, split=split)

    def _make_dataset(self, split):
        dataset_dir = osp.join(self.root, 'benchmark_RELEASE', 'dataset')
        if not osp.exists(dataset_dir):
            self.download()
        imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
            self.imgs.append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def loadlbl(self, lbl_file):
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls']['Segmentation'][0][0].astype(np.uint8)
        lbl = PIL.Image.fromarray(lbl)
        return lbl

    def download(self):
        import urllib3
        import shutil
        import tarfile

        filename = os.path.join(self.root, 'benchmark.tgz')

        http = urllib3.PoolManager()
        with http.request('GET', self.url, preload_content=False) as r, open(filename, 'wb') as out_file:
            print('==> Downloading SBD dataset from {}'.format(self.url))
            shutil.copyfileobj(r, out_file)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(filename, 'r')
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
