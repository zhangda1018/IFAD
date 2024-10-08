# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, mode, cfg, root = './datasets/imbalance_cifar10', imb_type='exp',
                 transform=None, target_transform=None, download=True):
        train = True if mode == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.cfg = cfg
        self.train = train
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            imb_factor = self.cfg.DATASET.IMBALANCECIFAR.RATIO
            # 获取不平衡数据集每个类的数量 [1000, 500, 250, 125, 62, 31, 15, 7, 3, 1]
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            # 接着从平衡数据集中获取不平衡数据集中的数据
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    # 获取不平衡数据集，这个并不针对于某个数据集
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        """
        根据给定的标注，创建一个字典，键是类别 id ，值是属于该类别的样本的索引列表。这样可以方便地根据类别来获取样本。
        {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6, 7, 8, 9]}
        """
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        """
        返回类别权重和总权重，类别权重和类别数量成反比
        """
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if  self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index], self.targets[index]
        meta = dict()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.data[sample_index], self.targets[sample_index]
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, meta

    # def my_get_item(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:  # 如果是训练模式，并且使用了加权采样
    #         assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]  # 检查采样类型是否合法
    #         if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":  # 如果采样类型是平衡
    #             sample_class = random.randint(0, self.cls_num - 1)  # 随机选择一个类别
    #         elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":  # 如果采样类型是反向
    #             sample_class = self.sample_class_index_by_weight()  # 根据权重随机选择一个类别
    #         sample_indexes = self.class_dict[sample_class]  # 获取该类别对应的索引列表
    #         index = random.choice(sample_indexes)  # 从索引列表中随机选择一个索引
    #
    #     img, target = self.data[index], self.targets[index]  # 根据索引，从数据集中获取图像和标签
    #     meta = dict()  # 初始化一个空字典，用来存储元数据
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)  # 将图像从数组转换为 PIL.Image 类型
    #
    #     if self.dual_sample:  # 如果使用了双重采样
    #         if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":  # 如果采样类型是反向
    #             sample_class = self.sample_class_index_by_weight()  # 根据权重随机选择一个类别
    #             sample_indexes = self.class_dict[sample_class]  # 获取该类别对应的索引列表
    #             sample_index = random.choice(sample_indexes)  # 从索引列表中随机选择一个索引
    #         elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":  # 如果采样类型是平衡
    #             sample_class = random.randint(0, self.cls_num - 1)  # 随机选择一个类别
    #             sample_indexes = self.class_dict[sample_class]  # 获取该类别对应的索引列表
    #             sample_index = random.choice(sample_indexes)  # 从索引列表中随机选择一个索引
    #         elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":  # 如果采样类型是均匀
    #             sample_index = random.randint(0, self.__len__() - 1)  # 随机选择一个索引
    #
    #         sample_img, sample_label = self.data[sample_index], self.targets[sample_index]  # 根据索引，从数据集中获取另一个图像和标签
    #         sample_img = Image.fromarray(sample_img)  # 将另一个图像从数组转换为 PIL.Image 类型
    #         sample_img = self.transform(sample_img)  # 对另一个图像进行变换
    #
    #         meta['sample_image'] = sample_img  # 将另一个图像添加到 meta 字典中，作为 sample_image 的键值对
    #         meta['sample_label'] = sample_label  # 将另一个标签添加到 meta 字典中，作为 sample_label 的键值对
    #
    #     return img, target, meta  # 返回图像、标签和 meta 字典组成的元组

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    # [{‘category_id’: 1}, {‘category_id’: 2}, {‘category_id’: 3}]
    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
