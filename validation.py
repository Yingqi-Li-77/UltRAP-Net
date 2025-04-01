"""
Date: April-1-2025
Author: Yingqi
"""

import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import logging
import sys
import random

from model.unet_network import UNet, ImgReconstructor
from model.ultra_model import UltraDecoderNMapModel
from mydataset import MixImgTestDataset
from config import parse_args, print_args


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(args):
    # 0. logging setting
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    logger.addHandler(chlr)

    model_stored_path = './TrainingRecord/' + args.retrain_net_path
    if not os.path.exists(model_stored_path):
        logger.info('trained model does not exist!')
        sys.exit(0)
    else:
        logger.info('trained model has been found!')

    # 1. model
    unet = UNet(in_channel=args.et_size, final_out_channel=args.final_out_channel, conv_kernel=args.conv_kernel)
    reconstruct_net = ImgReconstructor(scale_factor=1.0, set_size=1, kernel=args.reconstruct_conv_kernel)
    render_model = UltraDecoderNMapModel(constant_net=unet, inconstant_net=reconstruct_net, args=args).to(args.device)

    if args.ckp_flag:
        ckp_path = os.path.join(args.train_net_path, args.ckp_path)
        ckp_state = torch.load(ckp_path, map_location=args.device)
        render_model.load_state_dict(ckp_state['model_state_dict'])

    # 2. test
    map_ssim_list = []
    with torch.no_grad():
        for group_i in range(0, 2):
            group_name = os.listdir(args.test_data_path)[group_i]
            img_dataset = MixImgTestDataset(args.test_data_path, args.set_size, group_name, args)
            img_dataloader = DataLoader(img_dataset, batch_size=1)
            for idx, (x, label) in enumerate(img_dataloader):
                output_dict = render_model(x, label)
                overall_loss = output_dict['overall_loss']
                syn_ssim = output_dict['syn_ssim']
                syn_mse = output_dict['syn_mse']
                individual_ssim = output_dict['individual_ssim']
                save_ssim_i = np.append(individual_ssim, 1-syn_ssim.cpu().numpy())
                logger.info('Group: %s, overall_loss: %g, syn_ssim: %g, syn_mse: %g', group_name, overall_loss.item(),
                            1. - syn_ssim.item(), syn_mse.item())
                map_ssim_list.append(save_ssim_i)


if __name__ == "__main__":
    fix_seed(64)
    args = parse_args()
    print_args()
    validate(args)
