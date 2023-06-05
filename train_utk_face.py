import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from flac import flac_loss
from datasets.utk_face import get_utk_face
from models.resnet import ResNet18
from utils.logging import set_logging
from utils.utils import (
    AverageMeter,
    MultiDimAverageMeter,
    accuracy,
    load_model,
    pretty_dict,
    save_model,
    set_seed,
)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="race")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--bs", type=int, default=128, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1000)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    return opt


def set_model(opt):
    model = ResNet18().cuda()
    criterion1 = nn.CrossEntropyLoss()
    protected_net = ResNet18()
    if opt.task == "race":
        protected_attr_model = "./bias_capturing_classifiers/bcc_race.pth"
    elif opt.task == "age":
        protected_attr_model = "./bias_capturing_classifiers/bcc_age.pth"
    dd = load_model(protected_attr_model)
    protected_net.load_state_dict(dd)
    protected_net.cuda()
    return model, criterion1, protected_net


def train(train_loader, model, criterion, optimizer, protected_net, opt):
    model.train()
    avg_loss = AverageMeter()
    avg_clloss = AverageMeter()
    avg_miloss = AverageMeter()
    total_b_pred = 0
    total = 0
    train_iter = iter(train_loader)
    total_steps = len(train_iter)
    for idx, (images, labels, biases, _) in enumerate(train_iter):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()

        images = images.cuda()
        logits, features = model(images)
        with torch.no_grad():
            pr_l, pr_feat = protected_net(images)
            predicted_race = pr_l.argmin(dim=1, keepdim=True)

        predicted_race = predicted_race.T

        loss_mi_div = opt.alpha * (flac_loss(pr_feat, features, labels))
        loss_cl = 0.01 * criterion(logits, labels)
        loss = loss_cl + loss_mi_div

        avg_loss.update(loss.item(), bsz)
        avg_clloss.update(loss_cl.item(), bsz)
        avg_miloss.update(loss_mi_div.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_b_pred += predicted_race.eq(biases.view_as(predicted_race)).sum().item()
        total += bsz
    return avg_loss.avg, avg_clloss.avg, avg_miloss.avg


def validate(val_loader, model):
    model.eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for idx, (images, labels, biases, _) in enumerate(val_loader):
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            output, _ = model(images)
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            (acc1,) = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(
                corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1)
            )

    return top1.avg, attrwise_acc_meter.get_mean()


def main():
    opt = parse_option()

    exp_name = f"flac-utk_face_{opt.task}-{opt.exp_name}-lr{opt.lr}-alpha{opt.alpha}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f"results/{exp_name}"
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, "INFO", str(save_path))
    logging.info(f"Set seed: {opt.seed}")
    set_seed(opt.seed)
    logging.info(f"save_path: {save_path}")

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = "../data/utk_face"
    train_loader = get_utk_face(
        root,
        batch_size=opt.bs,
        bias_attr=opt.task,
        split="train",
        aug=False,
    )

    val_loaders = {}
    val_loaders["valid"] = get_utk_face(
        root, batch_size=256, bias_attr=opt.task, split="valid", aug=False
    )

    val_loaders["test"] = get_utk_face(
        root, batch_size=256, bias_attr=opt.task, split="test", aug=False
    )

    model, criterion, protected_net = set_model(opt)

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decay_epochs, gamma=0.1
    )

    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_accs = {"valid": 0, "test": 0}
    best_epochs = {"valid": 0, "test": 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(
            f"[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}"
        )
        loss, cllossp, milossp = train(
            train_loader, model, criterion, optimizer, protected_net, opt
        )
        logging.info(
            f"[{epoch} / {opt.epochs}] Loss: {loss}  Loss CE: {cllossp}  Loss MI: {milossp}"
        )

        scheduler.step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            accs, valid_attrwise_accs = validate(val_loader, model)

            stats[f"{key}/acc"] = accs.item()
            stats[f"{key}/acc_unbiased"] = torch.mean(valid_attrwise_accs).item() * 100
            eye_tsr = torch.eye(2)
            stats[f"{key}/acc_skew"] = (
                valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100
            )
            stats[f"{key}/acc_align"] = (
                valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
            )

        logging.info(f"[{epoch} / {opt.epochs}] {valid_attrwise_accs} {stats}")
        for tag in val_loaders.keys():
            if stats[f"{tag}/acc_unbiased"] > best_accs[tag]:
                best_accs[tag] = stats[f"{tag}/acc_unbiased"]
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(
                    **{f"best_{tag}_{k}": v for k, v in stats.items()}
                )

                save_file = save_path / "checkpoints" / f"best_{tag}.pth"
                save_model(model, optimizer, opt, epoch, save_file)
            logging.info(
                f"[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}"
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Total training time: {total_time_str}")

    save_file = save_path / "checkpoints" / f"last.pth"
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == "__main__":
    main()
