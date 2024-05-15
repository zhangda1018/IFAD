import numpy as np
import torch, math
from lib.core.evaluate import accuracy

class Combiner:
    def __init__(self, cfg, device, num_classes):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE # bbn_mix
        self.device = device
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

        # adapt reweighting
        self.num_classes = num_classes
        self.loss_type = cfg.LOSS.LOSS_TYPE
        self.weight = torch.ones(self.num_classes)

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def update_weight(self, out_0, out_1):
        var_0 = torch.var(out_0)
        var_1 = torch.var(out_1)

        # l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # 余弦衰减
        l = 1 - (self.epoch-1) / self.div_epoch  # 线性衰减
        # l = math.exp(-5 * ((self.epoch - 1) / self.div_epoch)) # 指数衰减

        self.weight = l*torch.ones(self.num_classes) + (1-l)*torch.tensor([var_1, var_0])
        pass
    
    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def default(self, model, criterion, image, label, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc

    def bbn_mix(self, model, criterion, image, label, meta, **kwargs):

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        # l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # 抛物线衰减
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # 余弦衰减
        l = 1 - (self.epoch-1) / self.div_epoch  # 线性衰减
        # l = math.exp(-5 * ((self.epoch - 1) / self.div_epoch)) # 指数衰减



        #l = 0.5  # fix
        #l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        #l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        #l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        #l = np.random.beta(self.alpha, self.alpha) # beta distribution
        #l = 1 if self.epoch <= 120 else 0  # seperated stage

        # mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
        # output = model(mixed_feature, classifier_flag=True)

        output_a = model(feature_a, label=label_a, device=self.device, classifier_flag=True, classifier_a=True)
        output_b = model(feature_b, label=label_b, device=self.device, classifier_flag=True, classifier_b=True)
        output = l*output_a + (1-l)*output_b # 是否要*2，做实验

        if self.loss_type == "VBLoss":
            loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b, self.weight.to(self.device))
        else:
            loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc

