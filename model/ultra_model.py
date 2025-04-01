"""
April-1-2025, Yingqi

This code is to construct the overall UltRAP-Net:

input image sequences --> U-shape extractor --> alpha-phi + p maps --> Ultrasound simulator --> Simulated images
                                                           |
                                                           |
                            Image Reconstructor <-----------
"""
import torch.nn
import torch.nn as nn
from copy import deepcopy

from render_fn import batch_render_method_convolutional_ultrasound
from ssim import ssim


class UltraDecoderNMapModel(nn.Module):
    def __init__(self, constant_net, inconstant_net, args, lindisp=False):
        super(UltraDecoderNMapModel, self).__init__()
        self.constant_net = constant_net
        self.inconstant_net = inconstant_net
        self.height = args.height
        self.width = args.width
        self.near = args.near
        self.far = args.far
        self.device = args.device
        self.batch_size = args.batch_size
        self.coef = args.loss_coef
        self.set_size = args.set_size
        self.lindisp = lindisp
        self.z_vals = self.get_z_vals()
        self.z_vals_test = torch.unsqueeze(self.z_vals[0], 0)

    def forward(self, input_imgs, label_imgs):
        """
        :param input_imgs: [batch_size, set_num, H_in, W_in]
        :param label_imgs: [batch_size, set_num, H_out, W_out]
        :return:
        """
        predicted_maps = self.constant_net(input_imgs)  # predicted_maps shape : [batch_size, 5+set_num, H_out, W_out]
        constant_maps = predicted_maps[:, 0:5, :, :]
        inconstant_maps = predicted_maps[:, 5:, :, :]

        # input the constant maps into the simulator and get the loss with the label image
        rendered_dict = batch_render_method_convolutional_ultrasound(constant_maps, self.z_vals)
        synthesized_img = rendered_dict['intensity_map'][:, None, :, :]
        rendered_dict['inconstant_map'] = inconstant_maps
        rendered_dict['label'] = label_imgs
        syn_mse = self.mse_loss(synthesized_img, label_imgs)
        syn_ssim = self.ssim_loss(synthesized_img, label_imgs)

        # input the constant maps and inconstant map into the decoder, and get the reconstructed images
        # note the reconstruction need to be conducted iteratively. In each iteration, 5+1 maps would be used
        reconstructed_imgs = []
        for i in range(self.set_size):
            reconstructed_img_i = self.inconstant_net(constant_maps,
                                                      torch.unsqueeze(inconstant_maps[:, i, :, :], dim=1))
            reconstructed_imgs.append(reconstructed_img_i)
        reconstructed_imgs = torch.cat(reconstructed_imgs, dim=1)
        recon_mse = self.mse_loss(input_imgs, reconstructed_imgs)
        recon_ssim = self.ssim_loss(input_imgs, reconstructed_imgs)

        # variation ranking loss
        vr_loss = torch.tensor(0., device=self.device)
        for i in range(self.batch_size):
            vr_loss_i = self.triplet_loss(torch.unsqueeze(input_imgs[i], dim=0),
                                          torch.unsqueeze(inconstant_maps[i], dim=0))
            vr_loss += vr_loss_i
        vr_loss = vr_loss / self.batch_size

        # map loss
        input_ren_imgs = torch.tile(synthesized_img, (1, self.set_size, 1, 1))
        predicted_maps_ren = self.constant_net(input_ren_imgs)
        constant_maps_ren = predicted_maps_ren[:, 0:5, :, :]
        render_dict_2 = batch_render_method_convolutional_ultrasound(constant_maps_ren, self.z_vals)
        render_dict_2['inconstant_map'] = predicted_maps_ren[:, 5:, :, :]

        # compare the two sets of maps
        map_us = self.get_maps_from_render_dict(rendered_dict)
        map_ren = self.get_maps_from_render_dict(render_dict_2)
        map_mse = self.mse_loss(map_us, map_ren)
        map_ssim = self.ssim_loss(map_us, map_ren)
        map_loss = (1 - self.coef) * 4 * map_mse + self.coef * map_ssim

        loss = 1. * ((1 - self.coef) * 4 * syn_mse + self.coef * syn_ssim) + (1 - self.coef) * 4 * recon_mse + \
               self.coef * recon_ssim + vr_loss + map_loss

        loss_dict = {'overall_loss': loss,
                     'syn_mse': syn_mse,
                     'syn_ssim': syn_ssim,
                     'recon_mse': recon_mse,
                     'recon_ssim': recon_ssim,
                     'recon_img': reconstructed_imgs,
                     'vr_loss': vr_loss,
                     'map_loss': map_loss,
                     'map_ssim': 1 + map_ssim,
                     'map_mse': map_mse,
                     'syn_img': synthesized_img,
                     'render_dict': rendered_dict,
                     }
        return loss_dict

    def val(self, input_imgs, label_imgs):
        """
        :param input_imgs: [batch_size, set_num, H_in, W_in]
        :param label_imgs: [batch_size, set_num, H_out, W_out]
        :return:
        """
        predicted_maps = self.constant_net(input_imgs)
        constant_maps = predicted_maps[:, 0:5, :, :]
        inconstant_maps = predicted_maps[:, 5:, :, :]

        # input the constant maps into the renderer and get the loss with the label image
        rendered_dict = batch_render_method_convolutional_ultrasound(constant_maps, self.z_vals)
        rendered_dict['inconstant_map'] = inconstant_maps
        rendered_dict['input'] = input_imgs
        rendered_dict['label'] = label_imgs
        rendered_dict['constant_map'] = constant_maps
        synthesized_img = rendered_dict['intensity_map'][:, None, :, :]
        syn_mse = self.mse_loss(synthesized_img, label_imgs)
        syn_ssim = self.ssim_loss(synthesized_img, label_imgs)

        # input the constant maps and inconstant map into the decoder, and get the reconstrucion loss
        # note the reconstruction need to be conducted using loop. In each loop, 5+1 maps would be used
        # label_imgs_for_recon = torch.tile(label_imgs, [1, self.set_size, 1, 1])
        reconstructed_imgs = []
        for i in range(self.set_size):
            reconstructed_img_i = self.inconstant_net(constant_maps,
                                                      torch.unsqueeze(inconstant_maps[:, i, :, :], dim=1))
            reconstructed_imgs.append(reconstructed_img_i)
        reconstructed_imgs = torch.cat(reconstructed_imgs, dim=1)
        rendered_dict['recon_img'] = reconstructed_imgs
        recon_mse = self.mse_loss(input_imgs, reconstructed_imgs)
        recon_ssim = self.ssim_loss(input_imgs, reconstructed_imgs)

        vr_loss = 0.
        for i in range(len(input_imgs)):
            vr_loss_i = self.triplet_loss(torch.unsqueeze(input_imgs[i], dim=0),
                                          torch.unsqueeze(inconstant_maps[i], dim=0))
            vr_loss += vr_loss_i
        vr_loss = vr_loss / self.batch_size

        loss = (1 - self.coef) * 4 * syn_mse + self.coef * syn_ssim + (1 - self.coef) * 4 * recon_mse + \
               self.coef * recon_ssim

        loss_dict = {'overall_loss': loss,
                     'syn_mse': syn_mse,
                     'syn_ssim': syn_ssim,
                     'recon_mse': recon_mse,
                     'recon_ssim': recon_ssim,
                     'vr_loss': vr_loss,
                     'syn_img': synthesized_img,
                     'recon_img': reconstructed_imgs,
                     'render_dict': rendered_dict,
                     'inconstant_map': inconstant_maps,
                     'label_img': label_imgs,
                     'input_img': input_imgs}
        return loss_dict

    def get_z_vals(self):
        """
        This function is to discretize the continuous ray into points
        input params:
        - n_samples: number of coarse samples per ray, default: H
        - near: default 0
        - far: probe_depth * scaling, default 0.014
        - n_rays: number of rays, default: W
        - lindisp:
        return:
        - z_vals [batch_size, W, H]
        """
        num_samples = deepcopy(self.height)
        num_rays = deepcopy(self.width)
        self.near = torch.tile(torch.tensor(self.near), [self.width, 1])
        self.far = torch.tile(torch.tensor(self.far), [self.width, 1])
        t_vals = torch.linspace(0., 1., num_samples)
        if not self.lindisp:
            # Space integration times linearly between 'near' and 'far'. Same
            # integration points will be used for all rays.
            z_vals = self.near * (1. - t_vals) + self.far * t_vals
        else:
            # Sample linearly in inverse depth (disparity).
            z_vals = 1. / (1. / self.near * (1. - t_vals) + 1. / self.far * t_vals)
        z_vals = torch.broadcast_to(z_vals, [self.batch_size, num_rays, num_samples]).to(self.device)
        return z_vals

    def mse_loss(self, predicted_x, label_x):
        loss_fn = torch.nn.MSELoss()
        mse_loss = loss_fn(predicted_x, label_x)
        return mse_loss

    def ssim_loss(self, predicted_x, label_x):
        mssim_loss = ssim(predicted_x, label_x)
        mssim_loss = 1. - mssim_loss
        return mssim_loss

    def triplet_loss(self, input_x, inconstant_map):
        anchor_idx = -1
        anchor_data = input_x[:, anchor_idx, :, :]
        data_diff = torch.sum(torch.square((input_x - anchor_data)), dim=(-1, -2))
        mask = torch.ones_like(data_diff, device=self.device)
        mask = torch.where(data_diff == torch.tensor(0.), torch.tensor(0., device=self.device), mask)
        data_diff = data_diff[mask == 1.]
        far_idx = torch.argmax(data_diff)
        near_idx = torch.argmin(data_diff)

        input_far, input_near = self.get_far_near_diff(input_x[:, anchor_idx], input_x[:, far_idx],
                                                       input_x[:, near_idx])
        feature_far, feature_near = self.get_far_near_diff(inconstant_map[:, anchor_idx], inconstant_map[:, far_idx],
                                                           inconstant_map[:, near_idx])
        feature_diff = feature_near - feature_far
        tanh_coeff = 100
        input_diff_tanh = torch.nn.functional.tanh(tanh_coeff * input_far) - torch.nn.functional.tanh(
            tanh_coeff * input_near)

        k = 1
        m = 1
        trip_loss = k * feature_diff + m * input_diff_tanh
        trip_loss = torch.nn.functional.relu(trip_loss)
        return trip_loss

    @staticmethod
    def get_maps_from_render_dict(render_dict):
        alpha = render_dict['attenuation_coeff']
        beta = render_dict['reflection_coeff']
        rho_b = render_dict['border probability']
        rho_s = render_dict['scatterers_density_coeff']
        phi = render_dict['scatter_amplitude']
        maps = torch.stack([alpha, beta, rho_b, rho_s, phi], dim=1)
        return maps

    @staticmethod
    def get_far_near_diff(anchor, far, near):
        far_diff = torch.nn.functional.mse_loss(anchor, far)
        near_diff = torch.nn.functional.mse_loss(anchor, near)
        return far_diff, near_diff
