"""
Date: April-1-2025
Author: Yingqi
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import logging
import random

from config import parse_args, print_args
from model.unet_network import UNet, ImgReconstructor
from model.ultra_model import UltraDecoderNMapModel
from mydataset import get_random_group, MixImgDataset, MixImgTestDataset


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    model_stored_path = os.path.join('TrainingRecord', args.train_net_path)
    if not os.path.exists(model_stored_path):
        os.makedirs(model_stored_path)
    if not os.path.exists(os.path.join('TrainingRecord', args.train_net_path)):
        os.makedirs(os.path.join('TrainingRecord', args.train_net_path))

    # 1. logging setting
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    logger.addHandler(chlr)

    # 2. writer setting
    writer = SummaryWriter(log_dir='SWrecord/' + args.retrain_net_path)

    # 3. network & render model
    unet = UNet(in_channel=args.set_size, final_out_channel=args.final_out_channel, conv_kernel=args.conv_kernel)
    reconstruct_net = ImgReconstructor(scale_factor=1.0, set_size=1, kernel=args.reconstruct_conv_kernel)
    render_model = UltraDecoderNMapModel(constant_net=unet, inconstant_net=reconstruct_net, args=args).to(args.device)

    # 4. optimizer
    optimizer = torch.optim.Adam(render_model.parameters(), args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_scheduler_gamma)

    # 5. load checkpoint
    if args.ckp_flag:
        ckp_path = os.path.join(model_stored_path, args.ckp_path)
        ckp_state = torch.load(ckp_path)
        render_model.load_state_dict(ckp_state['model_state_dict'])
        optimizer.load_state_dict(ckp_state['optimizer_state_dict'])
        step = ckp_state['step'] + 1
    else:
        step = 0

    # 6. train and test
    for epoch in range(args.num_epoch):
        group_name = get_random_group(path_folder=args.train_data_path)
        mydataset = MixImgDataset(args.train_data_path, args.set_size, group_name, args)
        img_dataloader = DataLoader(mydataset, args.batch_size, drop_last=True)
        for idx, (x, label) in enumerate(img_dataloader):
            loss_dict = render_model(x, label)
            loss = loss_dict['overall_loss']
            syn_mse = loss_dict['syn_mse']
            syn_ssim = loss_dict['syn_ssim']
            recon_mse = loss_dict['recon_mse']
            recon_ssim = loss_dict['recon_ssim']
            vr_loss = loss_dict['vr_loss']
            map_loss = loss_dict['map_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                logger.info('Step %d: Train Loss %g', step, loss.item())
            writer.add_scalars('Overall Loss', {'Train': loss.item()}, step)
            writer.add_scalars('Synth MSE Loss', {'Train': syn_mse.item()}, step)
            writer.add_scalars('Synth SSIM Loss', {'Train': syn_ssim.item()}, step)
            writer.add_scalars('Recon MSE Loss', {'Train': recon_mse.item()}, step)
            writer.add_scalars('Recon SSIM Loss', {'Train': recon_ssim.item()}, step)
            writer.add_scalars('Variation Ranking Loss', {'Train': vr_loss.item()}, step)
            writer.add_scalars('Map Loss', {'Train': map_loss.item()}, step)
            writer.add_scalars('Learning Rate', {'Train': optimizer.param_groups[0]['lr']}, step)
            if step % 2000 == 0 and step != 0:
                checkpoint = {
                    'model_state_dict': render_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step
                }
                path_checkpoint = os.path.join('TrainingRecord', args.retrain_net_path,
                                               'checkpoint_{}_step.pt'.format(step))
                os.makedirs(os.path.join('TrainingRecord', args.retrain_net_path), exist_ok=True)
                torch.save(checkpoint, path_checkpoint)

            if step % 100 == 0:
                with torch.no_grad():
                    test_group = get_random_group(args.test_data_path)
                    test_dataset = MixImgTestDataset(args.test_data_path, args.set_size, test_group, args)
                    test_dataloader = DataLoader(test_dataset, args.batch_size, drop_last=True)
                    for idx_test, (x_test, y_test) in enumerate(test_dataloader):
                        loss_dict_test = render_model(x_test, y_test)
                        overall_loss_test = loss_dict_test['overall_loss']
                        syn_mse_test = loss_dict_test['syn_mse']
                        syn_ssim_test = loss_dict_test['syn_ssim']
                        recon_mse_test = loss_dict_test['recon_mse']
                        recon_ssim_test = loss_dict_test['recon_ssim']
                        trip_loss_test = loss_dict_test['vr_loss']
                        map_loss_test = loss_dict_test['map_loss']

                        logger.info('Step %d: Test Loss %g', step, overall_loss_test.item())
                        writer.add_scalars('Test Loss', {'Test': overall_loss_test.item()}, step)
                        writer.add_scalars('Test Synth MSE Loss', {'Test': syn_mse_test.item()}, step)
                        writer.add_scalars('Test Synth SSIM Loss', {'Test': syn_ssim_test.item()}, step)
                        writer.add_scalars('Test Triplet Loss', {'Test': trip_loss_test.item()}, step)
                        writer.add_scalars('Test Reconstruction MSE Loss', {'Test': recon_mse_test.item()}, step)
                        writer.add_scalars('Test Reconstruction SSIM Loss', {'Test': recon_ssim_test.item()}, step)
                        writer.add_scalars('Test Map Loss', {'Test': map_loss_test.item()}, step)
            step += 1


if __name__ == "__main__":
    fix_seed(64)
    args = parse_args()
    print_args()
    train(args)
