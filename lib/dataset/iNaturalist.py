from lib.dataset.baseset import BaseSet
import random, cv2
from lib.utils.utils import prob_to_weight
import math


class iNaturalist(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)
        # random.seed(0)
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode=="train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

            self.sample_reverse_weight_dict = self.get_sample_reverse_weight(self.data, self.num_classes)
            self.sample_adpt_weight_dict = self.get_sample_adpt_weight()
            self.epoch_number = cfg.TRAIN.MAX_EPOCH
            self.epoch = 0
            self.initilize_all_parameters()

    def initilize_all_parameters(self):
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def get_sample_reverse_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
        max_num = max(num_list)
        class_probability = [i / max_num for i in num_list]

        sample_weight_dict = dict()
        for item in annotations:
            sample_weight_dict[item["image_id"]] = class_probability[item["category_id"]]
        return prob_to_weight(sample_weight_dict)

    def get_sample_adpt_weight(self):
        sample_weight_dict = dict()
        for item in self.data:
            sample_weight_dict[item["image_id"]] = 1
        return sample_weight_dict
    
    def update_sample_weight(self, sample_dict):
        self.sample_adpt_weight_dict.update(sample_dict)
        pass

    def update_epoch(self, epoch):
        self.epoch = epoch

    def get_sample_index_by_weight(self, mixed_weight):
        sum_weight = sum(mixed_weight)
        rand_number, now_sum = random.random() * sum_weight, 0
        for i in range(len(mixed_weight)):
            now_sum += mixed_weight[i]
            if rand_number <= now_sum:
                return i


    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == 'train':
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if  self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)

        meta = dict()
        if self.dual_sample:
            # ====================================================
            # reverse：根据权重选择一个类别，然后随机选择类别中的样本
            # balance：随机选择一个类别，然后随机选择类别中的样本
            # adpt：更新自适应权重，根据权重选择一个类别，根据权重选择一个样本
            # ====================================================
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)               
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "adpt":
                # 采样自适应权衡参数
                # l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
                # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # 余弦衰减
                l = 1 - (self.epoch-1) / self.div_epoch  # 线性衰减
                # l = math.exp(-5 * ((self.epoch - 1) / self.div_epoch)) # 指数衰减

                # 提取两个权重list
                key_list = list()
                reverse_list = list()
                adpt_list = list()
                for key in self.sample_reverse_weight_dict:
                    assert key in self.sample_adpt_weight_dict
                    key_list.append(key)
                    reverse_list.append(self.sample_reverse_weight_dict[key])
                    adpt_list.append(self.sample_adpt_weight_dict[key])

                # 计算混合权重
                mixed_weight = [(l * reverse_list[i] + (1-l) * adpt_list[i]) for i in range(len(reverse_list))]
                # _index实际上给出所选择的样本在mixed_weight中的位置，需要根据这个位置找到key值，即image_id
                _index = self.get_sample_index_by_weight(mixed_weight)
                sample_index = key_list[_index] - 1 # 与class_dict中的数据对齐

            sample_info = self.data[sample_index]
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id']
            sample_img = self.transform(sample_img)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.mode != 'test':
            image_label = now_info['category_id']  # 0-index

        return image, image_label, meta

# 设置一个采样集，数据为训练数据，在验证时计算每个样本的置信度
class Sample_Weight(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)
        # random.seed(0)


    def __getitem__(self, index):

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        if self.mode != 'test':
            image_label = now_info['category_id']  # 0-index

        return image, image_label, now_info["image_id"]









