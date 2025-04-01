import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_net_path', type=str, help='trained model stored path',
                        default='train_5+n_map_arm_overall_map_loss_revised_vrloss')
    parser.add_argument('--retrain_net_path', type=str, help='retrained model stored path',
                        default='train_5+n_map_arm_overall_map_loss_revised_vrloss')
    parser.add_argument('--ckp_path', type=str, help='path of model to be loaded', default='checkpoint_120000_step.pt')
    parser.add_argument('--train_data_path', type=str, help='training dataset stored path', default='data/training_set')
    parser.add_argument('--test_data_path', type=str, help='test dataset stored path', default='data/test_set')
    parser.add_argument('--label_data_path', type=str, help='label image stored path', default='data/label')

    parser.add_argument('--ckp_flag', action='store_true', help='whether to load checkpoint')
    parser.add_argument('--train_flag', action='store_false', help='whether to train')
    parser.add_argument('--test_flag', action='store_false', help='whether to test')

    parser.add_argument('--lr', type=float, help='init learning rate', default=1e-4)
    parser.add_argument('--lr_scheduler_gamma', type=float, help='learning rate scheduler',
                        default=0.5 ** (1 / (1 * 1e4)))
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--set_size', type=int, help='number of slices in one-shot input', default=4)
    parser.add_argument('--num_epoch', type=int, help='total epoch number', default=1000)
    parser.add_argument('--loss_coef', type=float, help='weight between mse and ssim losses', default=0.7)
    parser.add_argument('--height', type=int, help='height of resized input image', default=512)
    parser.add_argument('--width', type=int, help='width of resized input image', default=512)
    parser.add_argument('--final_out_channel', type=int, help='number of predicted maps', default=9)
    parser.add_argument('--conv_kernel', type=int, default=3)
    parser.add_argument('--reconstruct_conv_kernel', type=int, default=[3, 3, 3])
    parser.add_argument('--near', type=float, help='discretization param in ultrasound simulator', default=0.)
    parser.add_argument('--far', type=float, help='probe depth * scaling', default=0.14)
    parser.add_argument('--left_corner_x_label', type=int, help='x coordinate of left corner of label image',
                        default=54)
    parser.add_argument('--left_corner_y_label', type=int, help='y coordinate of left corner of label image',
                        default=91)
    parser.add_argument('--height_label', type=int, help='height of label image', default=400)
    parser.add_argument('--width_label', type=int, help='width of label image', default=400)
    parser.add_argument('--left_corner_x_input', type=int, help='x coordinate of left corner of input image',
                        default=54)
    parser.add_argument('--left_corner_y_input', type=int, help='y coordinate of left corner of input image',
                        default=91)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def print_args():
    args = parse_args()
    print("Training hyperparameters of UltRAP-Net:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


if __name__ == "__main__":
    print_args()
