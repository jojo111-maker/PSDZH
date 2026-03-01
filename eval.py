import glob

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from _data_zs import build_loaders, get_class_num, get_topk, get_atts
from _utils import init, print_in_md, mean_average_precision
from config import get_config
from network import build_model


def predict(net, dataloader, atts):
    device = next(net.parameters()).device

    probs, codes, clses = [], [], []
    net.eval()

    for imgs, labs, _ in tqdm(dataloader, desc=f"Extracting features"):
        with torch.no_grad():
            out = net(imgs.to(device), atts=atts)

        probs.append(out[-1])
        codes.append(out[0])
        clses.append(labs)
    return torch.cat(probs), torch.cat(codes).sign(), torch.cat(clses).to(device)


def calc_map(qB, rB, qL, rL, rP, seen_indices, topk=None, gamma=0.4):
    N, K = qB.shape

    if topk is None:
        topk = rL.shape[0]

    # Retrieve images from database
    if qL.ndim == 1:
        masks = qL.unsqueeze(1) == rL.unsqueeze(0)
    elif qL.ndim == 2:
        masks = qL @ rL.T > 0
    else:
        raise NotImplementedError(f"Not support: {qL.shape}")

    # Compute hamming distance
    hamming_dists = 0.5 * (K - qB @ rB.T)

    # Calibrate dists
    probs = F.softmax(rP, dim=1)
    probs[:, seen_indices] -= gamma
    pred_labels = probs.argmax(dim=1)
    pred_seen_idxes = torch.isin(pred_labels, seen_indices).nonzero(as_tuple=True)[0]

    hamming_dists[:, pred_seen_idxes] = K + 1

    ap_sum = 0.0
    for i in range(N):
        # Calculate hamming distance
        hamming_dist = hamming_dists[i]

        # Arrange position according to hamming distance
        topk_indices = torch.topk(hamming_dist, topk, dim=0, largest=False).indices
        retrieval = masks[i, topk_indices]

        # Get rank (1-based index) in all samples
        rank_in_all = retrieval.nonzero(as_tuple=True)[0] + 1
        if rank_in_all.numel() == 0:
            # Can not retrieve images
            continue

        # Generate rank (1-based index) in positive samples
        rank_in_pos = torch.arange(1, rank_in_all.numel() + 1, device=rank_in_all.device)  # way2

        ap_sum += (rank_in_pos / rank_in_all).mean().item()

    return ap_sum / N


if __name__ == "__main__":
    init()

    args = get_config()
    args.device = "cuda:1"

    rst = []
    for dataset in ["awa2", "cub", "sun"]:
        logger.info(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )

        atts = get_atts(args.dataset, args.data_dir).to(args.device)

        x = train_loader.dataset.get_all_labels().sum(dim=0).to(args.device)
        seen_indices = x.nonzero(as_tuple=True)[0]

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [128]:
            logger.info(f"Processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            net, _ = build_model(args, False)

            pth_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            pkl_list = glob.glob(f"{pth_dir}/*.pth")
            if len(pkl_list) != 1:
                logger.error(pkl_list)
                raise Exception(f"Cannot locate one *.pth in: {pth_dir}")

            checkpoint = torch.load(pkl_list[0], map_location="cpu")
            msg = net.load_state_dict(checkpoint["model"])
            logger.info(f"Model loaded: {msg}")

            _, qB, qL = predict(net, query_loader, atts)
            rP, rB, rL = predict(net, dbase_loader, atts)

            map1 = mean_average_precision(qB, rB, qL, rL, args.topk)
            if map1 != checkpoint["map"]:
                logger.warning(f"{map1} != {checkpoint['map']}")

            map2 = calc_map(qB, rB, qL, rL, rP, seen_indices, topk=args.topk, gamma=0.8)

            print(f"{map1:.3f} → {map2:.3f}")

            # DIML way of calculating mAP is stored in best_epoch
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_map": f"{map1:.3f}→{map2:.3f}"})

    print_in_md(rst)
