import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datasets.imagenet import get_imagenet
from models.imagenet_models import resnet18
from utils.logging import set_logging
from utils.utils import AverageMeter, pretty_dict, save_model, set_seed
from flac import flac_loss
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
    )
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--print_freq", type=int, default=300, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="save frequency")
    parser.add_argument(
        "--epochs", type=int, default=250, help="number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1000)

    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)

    # hyperparameters
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--ratio", type=int, default=0)
    parser.add_argument("--aug", type=int, default=1)

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model():
    model = resnet18(num_classes=9).cuda()
    criterion = nn.CrossEntropyLoss()

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    avg_ce_loss = AverageMeter()
    avg_con_loss = AverageMeter()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    for idx, (images, labels, _, _, pr_feat) in enumerate(tqdm(train_iter)):
        bsz = labels.shape[0]

        labels = labels.cuda()
        pr_feat = pr_feat.cuda()
        images = images.cuda()
        logits, features = model(images)

        loss_mi_div = opt.alpha * (flac_loss(pr_feat, features, labels, 0.5))

        loss_cl = criterion(logits, labels)
        loss = loss_cl + loss_mi_div

        avg_ce_loss.update(loss_cl.item(), bsz)
        avg_con_loss.update(loss_mi_div.item(), bsz)
        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_ce_loss.avg, avg_con_loss.avg, avg_loss.avg


def imagenet_unbiased_accuracy(
    outputs, labels, cluster_labels, num_correct, num_instance, num_cluster_repeat=3
):
    for j in range(num_cluster_repeat):
        for i in range(outputs.size(0)):
            output = outputs[i]
            label = labels[i]
            cluster_label = cluster_labels[j][i]

            _, pred = output.topk(1, 0, largest=True, sorted=True)
            correct = pred.eq(label).view(-1).float()

            num_correct[j][label][cluster_label] += correct.item()
            num_instance[j][label][cluster_label] += 1

    return num_correct, num_instance


def n_correct(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    n_correct = (predicted == labels).sum().item()
    return n_correct


def validate(
    val_loader, model, num_classes=9, num_clusters=9, num_cluster_repeat=3, key=None
):
    model.eval()

    total = 0
    f_correct = 0
    num_correct = [
        np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)
    ]
    num_instance = [
        np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)
    ]

    for images, labels, bias_labels, index, _ in val_loader:
        images, labels = images.cuda(), labels.cuda()
        for bias_label in bias_labels:
            bias_label.cuda()

        output, _ = model(images)

        batch_size = labels.size(0)
        total += batch_size

        if key == "unbiased":
            num_correct, num_instance = imagenet_unbiased_accuracy(
                output.data,
                labels,
                bias_labels,
                num_correct,
                num_instance,
                num_cluster_repeat,
            )
        else:
            f_correct += n_correct(output, labels)

    if key == "unbiased":
        for k in range(num_cluster_repeat):
            x, y = [], []
            _num_correct, _num_instance = (
                num_correct[k].flatten(),
                num_instance[k].flatten(),
            )
            for i in range(_num_correct.shape[0]):
                __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                if __num_instance >= 10:
                    x.append(__num_instance)
                    y.append(__num_correct / __num_instance)
            f_correct += sum(y) / len(x)

        return f_correct / num_cluster_repeat
    else:
        return f_correct / total


def main():
    opt = parse_option()

    exp_name = f"flac-imagenet-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-alpha{opt.alpha}-ratio{opt.ratio}-aug{opt.aug}-seed{opt.seed}-epochs{opt.epochs}"
    opt.exp_name = exp_name

    output_dir = f"results/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    imagenet_path = "../../data/imagenet"
    imagenet_a_path = "../../data/imagenet-a"
    feat_root = "../imagenet_biased_feats"

    train_loader = get_imagenet(
        f"{imagenet_path}/train",
        f"{imagenet_path}/train",
        batch_size=opt.bs,
        train=True,
        bias_feature_root=feat_root,
        load_bias_feature=True,
        aug=1,
    )

    val_loaders = {}
    val_loaders["biased"] = get_imagenet(
        f"{imagenet_path}/val",
        f"{imagenet_path}/val",
        batch_size=128,
        train=False,
        aug=False,
    )
    val_loaders["unbiased"] = get_imagenet(
        f"{imagenet_path}/val",
        f"{imagenet_path}/val",
        batch_size=128,
        train=False,
        aug=False,
    )
    val_loaders["ImageNet-A"] = get_imagenet(
        imagenet_a_path,
        imagenet_a_path,
        batch_size=128,
        train=False,
        val_data="ImageNet-A",
    )

    model, criterion = set_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = pretty_dict(**{"unbiased": 0, "ImageNet-A": 0})
    best_epochs = pretty_dict(**{"unbiased": 0, "ImageNet-A": 0})
    best_stats = pretty_dict()
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(
            f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}"
        )
        ce_loss, con_loss, loss = train(
            train_loader, model, criterion, optimizer, epoch, opt
        )
        logging.info(
            f"[{epoch} / {opt.epochs}] Loss: {loss} CE Loss: {ce_loss} Con Loss: {con_loss}"
        )

        scheduler.step()

        stats = pretty_dict()
        for key, val_loader in val_loaders.items():
            val_acc = validate(val_loader, model, key=key)
            stats[f"valid/acc_{key}"] = val_acc

        logging.info(f"[{epoch} / {opt.epochs}] current: {stats}")

        for key in best_accs.keys():
            if stats[f"valid/acc_{key}"] > best_accs[key]:
                best_accs[key] = stats[f"valid/acc_{key}"]
                best_epochs[key] = epoch
                best_stats[key] = stats

                save_file = save_path / "checkpoints" / f"best_{key}.pth"
                save_model(model, optimizer, opt, epoch, save_file)
            logging.info(
                f"[{epoch} / {opt.epochs}] best {key} accuracy: {best_accs[key]} at epoch {best_epochs[key]} \n best_stats: {best_stats[key]}"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")

    save_file = save_path / "checkpoints" / f"last.pth"
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == "__main__":
    main()
