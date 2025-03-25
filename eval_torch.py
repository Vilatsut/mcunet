import os
from tqdm import tqdm
import json

import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from mcunet.tinynas.data_providers.aurora import AuroraDataProvider
from mcunet.utils import (
    AverageMeter,
    accuracy,
    count_net_flops,
    count_parameters,
    set_running_statistics,
)


# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument("--net_id", type=str, help="net id of the model")
# data loader setting
parser.add_argument("--dataset", default="vww", type=str, choices=["imagenet", "vww"])
parser.add_argument(
    "--data-dir",
    default=os.path.expanduser("data/"),
    help="path to ImageNet validation data",
)
parser.add_argument(
    "--batch-size", type=int, default=128, help="input batch size for training"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers",
)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = "cuda"


def build_val_data_loader(resolution):
    normalize = transforms.Normalize(mean=[0.23280394, 0.24616548, 0.26092353], std=[0.16994016, 0.17286949, 0.16250615])
    kwargs = {"num_workers": args.workers, "pin_memory": True}

    if args.dataset == "imagenet":
        val_transform = transforms.Compose(
            [
                transforms.Resize(int(resolution * 256 / 224)),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif args.dataset == "vww":
        val_transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution)
                ),  # if center crop, the person might be excluded
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    return val_loader


def calib_bn(net, resolution, batch_size, num_images=2000):
    # print('Creating dataloader for resetting BN running statistics...')
    normalize = transforms.Normalize(mean=[0.23280394, 0.24616548, 0.26092353], std=[0.16994016, 0.17286949, 0.16250615])
    dataset = datasets.ImageFolder(
        args.data_dir,
        transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    # chosen_indexes = np.random.choice(list(range(len(dataset))), num_images)
    # sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sub_sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    set_running_statistics(net, data_loader)


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc="Validate") as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, 1-target).item())
                top1 = accuracy(output, 1-target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({"loss": val_loss.avg, "top1": val_top1.avg})
                t.update(1)

    return val_top1.avg


def main():
    # 96, 128, 160
    """
    model, resolution, description = build_model(args.net_id, pretrained=True)
    model = model.to(device)
    model.eval()
    val_loader = build_val_data_loader(resolution)

    # profile model
    total_macs = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = count_parameters(model)
    print(
        " * FLOPs: {:.4}M, param: {:.4}M".format(total_macs / 1e6, total_params / 1e6)
    )

    acc = validate(model, val_loader)
    print(" * Accuracy: {:.2f}%".format(acc))
    """

    from mcunet.tinynas.elastic_nn.networks import OFAMCUNets

    ofa_network = OFAMCUNets(
        n_classes=2,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.0,
        base_stage_width="proxyless",
        width_mult_list=[1.3],
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    )
    resolution = 224

    ofa_network.load_state_dict(
        torch.load("pretrained/supernet.pth", map_location="cpu")["state_dict"], strict=True
    )

    # cfg = ofa_network.sample_active_subnet()
    cfg = {
        "wid": 0,
        "ks": [7, 5, 3, 5, 3, 3, 7, 7, 7, 3, 7, 5, 7, 7, 3, 7, 5, 7, 7, 5],
        "e": [3, 6, 4, 3, 4, 3, 6, 4, 3, 3, 6, 6, 4, 6, 6, 3, 3, 3, 6, 6],
        "d": [2, 1, 0, 1, 2, 2],
    }
    max_cfg = dict()
    max_cfg["wid"] = cfg["wid"]
    max_cfg["ks"] = [7 for _ in range(len(cfg["ks"]))]
    max_cfg["e"] = [6 for _ in range(len(cfg["e"]))]
    max_cfg["d"] = [2 for _ in range(len(cfg["d"]))]
    ofa_network.set_active_subnet(**max_cfg)
    # ofa_network.set_active_subnet(**cfg)
    subnet = ofa_network.get_active_subnet().to(device)
    calib_bn(subnet, resolution, batch_size=100)
    val_loader = build_val_data_loader(resolution)

    data_iter = iter(val_loader)
    images, labels = next(data_iter)  # Get a batch
    mean = torch.tensor([0.23280394, 0.24616548, 0.26092353]).view(3, 1, 1)  # Adjust based on dataset
    std = torch.tensor([0.16994016, 0.17286949, 0.16250615]).view(3, 1, 1)
    images = images * std + mean  # Undo normalization
    
    grid = make_grid(images[:16], nrow=4, normalize=False)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))  # Convert CHW to HWC format
    plt.axis('off')

    plt.text(
        x=(0 % 4) * 4 // 4 + 5,  # Positioning text based on grid
        y=(0 // 4) * 4 // 4 + 5,  # Positioning text based on grid
        s=str(labels[0].item()),  # Display label
        color='white',
        fontsize=40,
        weight='bold'
    )
    plt.show()
    # top1 = validate(subnet, val_loader)
    # print(top1)


if __name__ == "__main__":
    main()
