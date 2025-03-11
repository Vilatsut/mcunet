import os
from tqdm import tqdm
import json

import torch
import argparse

from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import build_val_data_loader, calib_bn, validate


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    # data loader setting
    parser.add_argument(
        "--resolution", default=160, type=int, choices=[32, 64, 96, 128, 144, 160]
    )
    parser.add_argument(
        "--supernet-dir",
        default=os.path.expanduser("vww_supernet.pth"),
        help="path to the supernet checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("data/"),
        help="path to ImageNet validation data",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("acc_datasets"),
        help="output dataset of the accuracy dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )

    args = parser.parse_args()

    device_id = torch.cuda.current_device()
    torch.cuda.set_device(device_id)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    ofa_network = OFAMCUNets(
        n_classes=2,
        bn_param=(0.1, 1e-5),
        dropout_rate=0.0,
        base_stage_width="proxyless",
        width_mult_list=[1.3],
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    )

    ofa_network.load_state_dict(
        torch.load(args.supernet_dir, map_location="cpu")["state_dict"], strict=True
    )

    ofa_network = ofa_network.to("cuda:%d" % device_id)

    all_results = []
    result_fn = os.path.join(
        args.output_dir,
        f"{ofa_network.__class__.__name__}_r{args.resolution}_gpu{device_id}_acc_table.json",
    )

    for i in tqdm(range(1250)):
        cfg = ofa_network.sample_active_subnet()
        ofa_network.set_active_subnet(**cfg)
        subnet = ofa_network.get_active_subnet().cuda()
        calib_bn(subnet, args.data_dir, args.batch_size, args.resolution)
        # important: split needs to be 1.
        val_loader = build_val_data_loader(
            args.data_dir, args.resolution, args.batch_size, split=1
        )
        acc = validate(subnet, val_loader)
        cfg["image_size"] = args.resolution
        all_results.append((cfg, acc))
        with open(result_fn, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
