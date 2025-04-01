import tensorflow as tf
import numpy as np
import torch

import sys
sys.path.append('..')


# TODO: Improve psf shape. Now, it is a 2D Gaussian kernel.

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    """
    pytorch version of "gaussian_kernel_tf"
    :param size:
    :param mean:
    :param std:
    :return:
    """
    delta_t = 1  # 9.197324e-01
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t
    d1 = torch.distributions.normal.Normal(mean, std * 2.)
    d2 = torch.distributions.normal.Normal(mean, std)
    tmp = torch.range(start=-size, end=(size), dtype=torch.float32) * delta_t
    vals_x = torch.exp(d1.log_prob(tmp))
    vals_y = torch.exp(d2.log_prob(tmp))
    gauss_kernel = torch.einsum('i,j->ij',
                                vals_x,
                                vals_y)

    return gauss_kernel / torch.sum(gauss_kernel)


def batch_render_method_convolutional_ultrasound(raw, z_vals, device):
    """
    pytorch version of "render_method_convolutional_ultrasound()", implementing in mini batch
    :param raw: [batch-size, channel, H, W], tf version: [W, H, channel]
    :param z_vals: [batch_size, W, H]
    :return:
    """

    g_size = 3
    g_mean = 0.
    g_variance = 1.
    g_kernel = gaussian_kernel(g_size, g_mean, g_variance)
    g_kernel = g_kernel[None, None, :, :].to(device)

    def raw2attenualtion(raw, dists):
        # yq: Beer-Lambert Law
        return torch.exp(-raw * dists)
    # transpose the raw into [batch_size, W, H, channel]
    raw = torch.permute(raw, [0, -2, -1, 1])
    # Compute distance between points In paper the points are sampled equidistantly
    # yq comments: the dist is obtained in height dim, previous H is dim=1, [W, 1], current H is dim=2, [batch_size, W, 1]
    dists = torch.abs(z_vals[:, :, :-1] - z_vals[:, :, 1:])
    # dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, :, -1, None]], dim=-1)
    # ATTENUATION
    # Predict attenuation coefficient for each sampled point. This value is positive.
    # yq comments: previous atten_coef is [W, H], current atten_coef is  [batch_size, W, H]
    attenuation_coeff = torch.abs(raw[:, :, :, 0])
    attenuation = raw2attenualtion(attenuation_coeff, dists)
    # Compute total attenuation at each pixel location as exp{-sum[a_n*d_n]}
    attenuation_transmission = torch.cumprod(attenuation, dim=-1)[:, :, :-1]
    attenuation_transmission = torch.cat((torch.ones(attenuation_transmission.size()[0],
                                                     attenuation_transmission.size()[1], 1).to(device),
                                          attenuation_transmission), dim=-1)
    # REFLECTION
    prob_border = torch.sigmoid(raw[:, :, :, 2])

    # Bernoulli distribution can be approximated by RelaxedBernoulli
    # temperature = 0.01
    # border_distribution = tf.contrib.distributions.RelaxedBernoulli(temperature, probs=prob_border)
    # Note: Estimating a border explicitly is not necessary. I recommend experimenting with solely relying on
    # reflection coefficient for the geometry estimation
    border_distribution = torch.distributions.bernoulli.Bernoulli(probs=prob_border)
    border_indicator = border_distribution.sample()

    border_indicator_np = np.array(border_indicator.cpu())
    # plt.imshow(np.squeeze(border_indicator_np), cmap='gray')
    # plt.show()

    # border_indicator.requires_grad = False
    # Predict reflection coefficient. This value is between (0, 1).
    reflection_coeff = torch.sigmoid(raw[:, :, :, 1])
    reflection_transmission = 1. - reflection_coeff * border_indicator
    reflection_transmission = torch.cumprod(reflection_transmission, dim=-1)[:, :, :-1]
    reflection_transmission = torch.cat((torch.ones(reflection_transmission.size()[0],
                                                    reflection_transmission.size()[1], 1).to(device),
                                         reflection_transmission), dim=-1)
    # reflection_transmission = torch.squeeze(reflection_transmission)
    border_convolution = torch.nn.functional.conv2d(input=border_indicator[:, None, :, :], weight=g_kernel, stride=1,
                                                    padding='same')
    border_convolution = torch.squeeze(border_convolution)

    # BACKSCATTERING
    # Scattering density coefficient can be either learned or constant for fully developed speckle
    density_coeff_value = torch.sigmoid(raw[:, :, :, 3])
    density_coeff = torch.ones_like(reflection_coeff).to(device) * density_coeff_value
    scatter_density_distibution = torch.distributions.bernoulli.Bernoulli(probs=density_coeff)
    scatterers_density = scatter_density_distibution.sample()
    # Predict scattering amplitude
    amplitude = torch.sigmoid(raw[:, :, :, 4])
    # Compute scattering template
    scatterers_map = torch.multiply(scatterers_density, amplitude)
    psf_scatter = torch.nn.functional.conv2d(input=scatterers_map[:, None, :, :], weight=g_kernel, stride=1,
                                             padding="same")
    psf_scatter = torch.squeeze(psf_scatter)
    # Compute remaining intensity at a point n
    transmission = torch.multiply(attenuation_transmission, reflection_transmission)
    # Compute backscattering part of the final echo
    b = torch.multiply(transmission, psf_scatter)
    # Compute reflection part of the final echo
    r = torch.multiply(torch.multiply(transmission, reflection_coeff), border_convolution)
    # Compute the final echo
    # Note: log compression has not been used for the submission
    # if args.log_compression:
    #     compression_constant = 3.14  # TODO: should be calculated based on r_reflection_maximum
    #     log_compression = lambda x: tf.math.log(1. + compression_constant * x) * tf.math.log(
    #         1. + compression_constant)
    #     r = log_compression(r)
    intensity_map = b + r
    ret = {'intensity_map': intensity_map,
           'attenuation_coeff': attenuation_coeff,
           'reflection_coeff': reflection_coeff,
           'attenuation_transmission': attenuation_transmission,
           'reflection_transmission': reflection_transmission,
           'scatterers_density': scatterers_density,
           'scatterers_density_coeff': density_coeff,
           'scatter_amplitude': amplitude,
           'border probability': prob_border,
           'b': b,
           'r': r,
           "transmission": transmission}
    return ret


def get_z_vals(N_samples, near, far, N_rays, lindisp=False):
    """

    :param N_samples: number of coarse samples per ray, default
    :param near: default 0
    :param far: probe_depth * scaling, default 0.014
    :param N_rays: batch_size?
    :param lindisp:
    :return:
    """
    t_vals = torch.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = torch.broadcast_to(z_vals, [N_rays, N_samples])
    return z_vals

