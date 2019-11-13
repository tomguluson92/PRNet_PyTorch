# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

from config.config import FLAGS

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

# Set random seem for reproducibility
manualSeed = 5
INFO("Random Seed", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def main(data_dir):
    # 0) Tensoboard Writer.
    writer = SummaryWriter(FLAGS['summary_path'])
    origin_img, uv_map_gt, uv_map_predicted = None, None, None

    if not os.path.exists(FLAGS['images']):
        os.mkdir(FLAGS['images'])

    # 1) Create Dataset of 300_WLP & Dataloader.
    wlp300 = PRNetDataset(root_dir=data_dir,
                          transform=transforms.Compose([ToTensor(),
                                                        ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])]))

    wlp300_dataloader = DataLoader(dataset=wlp300, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=4)

    # 2) Intermediate Processing.
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
    ])

    # 3) Create PRNet model.
    start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
    model = ResFCN256()

    # Load the pre-trained weight
    if FLAGS['resume'] and os.path.exists(os.path.join(FLAGS['images'], "latest.pth")):
        state = torch.load(os.path.join(FLAGS['images'], "latest.pth"))
        model.load_state_dict(state['prnet'])
        start_epoch = state['start_epoch']
        INFO("Load the pre-trained weight! Start from Epoch", start_epoch)
    else:
        start_epoch = 0
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS["lr"], betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    stat_loss = SSIM(mask_path=FLAGS["mask_path"], gauss=FLAGS["gauss_kernel"])
    loss = WeightMaskLoss(mask_path=FLAGS["mask_path"])

    for ep in range(start_epoch, target_epoch):
        bar = tqdm(wlp300_dataloader)
        Loss_list, Stat_list = [], []
        for i, sample in enumerate(bar):
            uv_map, origin = sample['uv_map'].to(FLAGS['device']), sample['origin'].to(FLAGS['device'])

            # Inference.
            uv_map_result = model(origin)

            # Loss & ssim stat.
            logit = loss(uv_map_result, uv_map)
            stat_logit = stat_loss(uv_map_result, uv_map)

            # Record Loss.
            Loss_list.append(logit.item())
            Stat_list.append(stat_logit.item())

            # Update.
            optimizer.zero_grad()
            logit.backward()
            optimizer.step()
            bar.set_description(" {} [Loss(Paper)] {} [SSIM({})] {}".format(ep, Loss_list[-1], FLAGS["gauss_kernel"], Stat_list[-1]))

            # Record Training information in Tensorboard.
            if origin_img is None and uv_map_gt is None:
                origin_img, uv_map_gt = origin, uv_map
            uv_map_predicted = uv_map_result

            writer.add_scalar("Original Loss", Loss_list[-1], FLAGS["summary_step"])
            writer.add_scalar("SSIM Loss", Stat_list[-1], FLAGS["summary_step"])

            grid_1, grid_2, grid_3 = make_grid(origin_img, normalize=True), make_grid(uv_map_gt), make_grid(uv_map_predicted)

            writer.add_image('original', grid_1, FLAGS["summary_step"])
            writer.add_image('gt_uv_map', grid_2, FLAGS["summary_step"])
            writer.add_image('predicted_uv_map', grid_3, FLAGS["summary_step"])
            writer.add_graph(model, uv_map)

        if ep % FLAGS["save_interval"] == 0:
            with torch.no_grad():
                origin = cv2.imread("./test_data/obama_origin.jpg")
                gt_uv_map = np.load("./test_data/test_obama.npy")
                origin, gt_uv_map = test_data_preprocess(origin), test_data_preprocess(gt_uv_map)

                origin, gt_uv_map = transform_img(origin), transform_img(gt_uv_map)

                origin_in = origin.unsqueeze_(0).cuda()
                pred_uv_map = model(origin_in).detach().cpu()

                save_image([origin.cpu(), gt_uv_map.unsqueeze_(0).cpu(), pred_uv_map],
                           os.path.join(FLAGS['images'], str(ep) + '.png'), nrow=1, normalize=True)

            # Save model
            state = {
                'prnet': model.state_dict(),
                'Loss': Loss_list,
                'start_epoch': ep,
            }
            torch.save(state, os.path.join(FLAGS['images'], 'latest.pth'))

            scheduler.step()

    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="specify input directory.")
    args = parser.parse_args()
    main(args.train_dir)
