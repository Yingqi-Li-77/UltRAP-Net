# UltRAP-Net: Reverse Approximation of Tissue Properties in Ultrasound Imaging

This repository includes the code and data of the _Advanced Intelligent Systems_ paper "UltRAP-Net: 
Reverse Approximation of Tissue Properties in Ultrasound Imaging"


 <!--https://github.com/user-attachments/assets/73efa5e6-43e9-4032-8e5a-7721c0fd6d16-->

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mSVIivfsRkw/0.jpg)](https://www.youtube.com/watch?v=mSVIivfsRkw)

## Links:
Paper: 
YouTube: https://www.youtube.com/watch?v=mSVIivfsRkw


## Abstract
Medical ultrasound (US) has been widely used in clinical practices due to its merits of being low-cost, real-time, and radiation-free.
However, its capability to reveal the underlying tissue properties has not yet been thoroughly studied. In this study, we propose a
learning-based framework to reversely approximate tissue property representations with physics-constrained. The shared property is
extracted from multiple B-mode images acquired with varying dynamic ranges. First, a feature extractor network is used to generate property maps, i.e., attenuation coefficient α, reflection coefficient β, border probability ρb, scattering density ρs and scattering
amplitude ϕ, capturing the shared physics across distinct inputs, and one perturbation (p) map characterizing the variations caused
by varying dynamic range. The α-ϕ maps are loosely regularized by rendering them forward to realistic US images using ray-tracing-based rendering. To further enforce the physics constraints, a ranking loss is introduced to align the disparity between two estimated
p maps with the discrepancy between two distinct inputs.

## Preparation

- Environment
```commandline
pip install -r requirements.txt
```


## Usage
- Dataset
  Download the dataset from this link:
  
     [https://drive.google.com/drive/folders/16LOMk92x2xzsNkYqW3tUWVXjHAuHOdda?usp=drive_link](https://drive.google.com/drive/folders/1woblVI9tw_KlqJbJUFuoCX3RwJkwPczS?usp=sharing)
  
  Labels are stored in `.\data\label`
- Training
  1. Set the hyperparameters in `config.py`.
  2. Run `training.py`.
- Testing
  1. Download the model checkpoint from this link:
     
     [https://drive.google.com/drive/folders/16LOMk92x2xzsNkYqW3tUWVXjHAuHOdda?usp=drive_link](https://drive.google.com/drive/folders/1YTdqEQP_9GlXI8ZTsbUkQAPTGMKf5l98?usp=sharing)
  3. Run `validation.py`.

## Citation
If you find our code or paper useful, please cite:

```commandline
The citation will be released when the paper production is completed.
```
