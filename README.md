# UltRAP-Net: Reverse Approximation of Tissue Properties in Ultrasound Imaging

This repository includes the code and data of the _Advanced Intelligent Systems_ paper "UltRAP-Net: 
Reverse Approximation of Tissue Properties in Ultrasound Imaging"



https://github.com/user-attachments/assets/30442edd-84b3-48d2-9b26-ea65b6db53af



## Abstract
Medical ultrasound (US) has been widely used in clinical practices due to its merits of being low-cost, real-time, and radiation-free.
However, its capability to reveal the underlying tissue properties has not yet been thoroughly studied. In this study, we propose a
learning-based framework to reversely approximate tissue property representations with physics-constrained. The shared property is
extracted from multiple B-mode images acquired with varying dynamic ranges. First, a feature extractor network is used to generate property maps, i.e., attenuation coefficient α, reflection coefficient β, border probability ρb, scattering density ρs and scattering
amplitude ϕ, capturing the shared physics across distinct inputs, and one perturbation (p) map characterizing the variations caused
by varying dynamic range. The α-ϕ maps are loosely regularized by rendering them forward to realistic US images using ray-tracingbased rendering. To further enforce the physics constraints, a ranking loss is introduced to align the disparity between two estimated
p maps with the discrepancy between two distinct inputs.

## Preparation

- Environment
```commandline
pip install -r requirements.txt
```


## Usage
- Dataset
- Training
  1. Set the hyperparameters in `config.py`.
  2. Run `training.py`.

## Citation
If you find our code or paper useful, please cite:

```commandline
The citation will be released when the paper production is completed.
```
