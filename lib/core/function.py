import main._init_paths
from lib.core.evaluate import accuracy, AverageMeter, FusionMatrix
from lib.utils.utils import prob_to_weight
from sklearn.metrics import f1_score

import numpy as np
import torch
import time


def train_model(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    combiner,
    criterion,
    cfg,
    logger,
    **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model_with_f1(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)

    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        all_preds = []
        all_labels = []
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature_a, feature_b = (
                model(image, feature_cb=True),
                model(image, feature_rb=True),
            )

            output_a = model(feature_a, label=None, device=device, classifier_flag=True, classifier_a=True)
            output_b = model(feature_b, label=None, device=device, classifier_flag=True, classifier_b=True)
            output = output_a + output_b

            loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

            all_preds.append(now_result.cpu().numpy())
            all_labels.append(label.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # f1 = f1_score(all_labels, all_preds, average='macro')
        f1_by_fm = fusion_matrix.get_f1_score()
        f1_minor = f1_by_fm[1]
        f1 = f1_minor

        if np.isnan(f1):
            f1 = 0

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}% F1_score:{:.4f}-------".format(
            epoch_number, all_loss.avg, acc.avg * 100, f1
        )

        logger.info(pbar_str)
    return acc.avg, all_loss.avg, f1


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)


    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)
            loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100
        )
        logger.info(pbar_str)
    return acc.avg, all_loss.avg


def sample_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()

    sample_dict = dict()

    with torch.no_grad():
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, image_id) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature_a, feature_b = (
                model(image, feature_cb=True),
                model(image, feature_rb=True),
            )

            output_a = model(feature_a, label=None, device=device, classifier_flag=True, classifier_a=True)
            output_b = model(feature_b, label=None, device=device, classifier_flag=True, classifier_b=True)
            output = output_a + output_b

            if i == 0:
                out_0 = output[label == 0][:,0]  # 选择标签为 0 的行
                out_1 = output[label == 1][:,1]  # 选择标签为 1 的行
            else:
                out_0 = torch.cat([out_0, output[label == 0][:,0]])
                out_1 = torch.cat([out_1, output[label == 1][:,1]])

            # 共享backbone权重，所以特征是相同的，需要修改output，使用原始分布的分类器
            # output = model(feature, classifier_flag=True)
            score_result = func(output)
            # print(score_result)

            # now_result返回最大值的索引，在这儿不需要索引，需要对应label位置的置信度
            now_result = torch.argmax(score_result, 1)
            label_values = score_result[range(len(label)), label]
            
            # print(image_id)
            # print(label_values)

            # 计算同一类别下所有样本的输出值，保存到list中
            # 在for循环结束之后计算方差，并更新权重，并且导入到crossEntropy中

            
            sample_dict.update({key: value for key, value in zip(image_id, label_values)})
            # print(sample_dict)

    # 计算的样本的置信度（概率），需要转换成权重
    sample_dict = {k: v.item() for k, v in sample_dict.items()}

    return prob_to_weight(sample_dict), out_0, out_1