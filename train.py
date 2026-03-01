import json
import os
import time

import torch
from loguru import logger
import torch.nn.functional as F
from _data_zs import build_loaders, get_topk, get_class_num, get_atts
from _utils import (
    AverageMeter,
    build_optimizer,
    calc_map_eval,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    rename_output,
    save_checkpoint,
    seed_everything,
    validate_smart,
)
from config import get_config
from loss import ZSLViTLoss, SCPLoss
from network import build_model


def train_epoch(args, dataloader, net, zsl_loss, scp_loss, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["L_pre", "L_SR", "L_VR", "L_SCP", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images = images.to(args.device)

        labels_idx = labels.argmax(1).to(args.device)  # class index，用于 cross-entropy
        labels_onehot = F.one_hot(labels_idx, num_classes=args.n_classes).float().to(args.device)  # one-hot，用于 SCP-Loss

        x, recon_v, recon_s, logits = net(images, labels=labels_idx, atts=globals()["atts"])

        L_pre, L_SR, L_VR = zsl_loss(x, recon_v, recon_s, logits, globals()["atts"], labels_idx)

        L_scp = scp_loss(x, labels_onehot)  # x 是 hash 特征, labels_onehot 用于 SCP

        # print(
        #     "DEBUG | "
        #     f"L_scp={L_scp.item():.6f}, "
        #     f"x_mean={x.abs().mean().item():.6f}, "
        #     f"label_sum={labels_onehot.sum().item():.1f}"
        # )

        stat_meters["L_pre"].update(L_pre.item())
        stat_meters["L_SR"].update(L_SR.item())
        stat_meters["L_VR"].update(L_VR.item())
        stat_meters["L_SCP"].update(L_scp.item())
        loss = (
                args.lambda1 * L_pre
                + args.lambda2 * L_SR
                + args.lambda3 * L_VR
                + args.lambda_scp * L_scp
        )

        stat_meters["loss"].update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        map_v = calc_map_eval(x.detach().sign(), labels_idx)
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setup net
    net, out_idx = build_model(args, True)

    # setup criterion
    # criterion = ZSLViTLoss(args)
    # criterion = ZSLViTLoss(args).to(device)
    zsl_loss = ZSLViTLoss().to(device)
    scp_loss = SCPLoss(args).to(device)

    logger.info(f"Number of net's params: {calc_learnable_params(net)}")

    # setup optimizer
    to_optim = [
        {
            "params": net.fc.parameters(),
            "lr": args.lr,
            "weight_decay": args.wd,
        },
        {
            "params": net.fc_list.parameters(),
            "lr": args.lr,
            "weight_decay": args.wd,
        },
        # {"params": net.mlp_g.parameters(), "lr": 100 * args.lr},
        {"params": net.mlp_g.parameters(), "lr": 0.005, "weight_decay": 1e-5},
        {
            "params": [
                p for name, p in net.named_parameters() if not name.startswith("fc") and not name.startswith("mlp_g")
            ],
            "lr": args.lr,
            "weight_decay": args.wd,
        },
    ]
    optimizer = build_optimizer(args.optimizer, to_optim)
    # optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, out_idx, zsl_loss, scp_loss, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, out_idx, zsl_loss, scp_loss, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, zsl_loss, scp_loss, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                out_idx=out_idx,
                multi_thread=args.multi_thread,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"Without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"Reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["cub", "sun", "awa2"]:
    # for dataset in ["sun"]:
    # for dataset in ["cub"]:
    # for dataset in ["awa2"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset,
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            samples_per_class=args.samples_per_class,
        )
        # args.n_samples = len(train_loader.dataset)
        globals()["atts"] = get_atts(args.dataset, args.data_dir).to(args.device)

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [16, 128]:
            print(f"Processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output-SR 0.2 VR 0.8/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass...")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()
