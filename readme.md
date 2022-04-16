[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-baselines-for-image-restoration/image-deblurring-on-gopro)](https://paperswithcode.com/sota/image-deblurring-on-gopro?p=simple-baselines-for-image-restoration)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-baselines-for-image-restoration/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=simple-baselines-for-image-restoration)

## NAFNet: Nonlinear Activation Free Network for Image Restoration

The official pytorch implementation of the paper **[Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)**

#### Liangyu Chen\*, Xiaojie Chu\*, Xiangyu Zhang, Jian Sun

>Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. 
>In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient. 
>To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are **not necessary**: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs.

![PSNR_vs_MACs](./figures/PSNR_vs_MACs.jpg)

### News
NAFNet based Stereo Image Super-Resolution solution won the **1st place** on the NTIRE 2022 Stereo Image Super-resolution Challenge! Coming Soon.

### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/megvii-research/NAFNet
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 
* Image Denoise Colab Demo: [<a href="https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)
* Image Deblur Colab Demo: [<a href="https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing)



### Results and Pre-trained Models

| name | Dataset|PSNR|SSIM| model (gdrive) | model (百度网盘) |
|:----|:----|:----|:----|:----|-----|
|NAFNet-GoPro-width32|GoPro|32.8705|0.9606|[link](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view?usp=sharing)|[link](https://pan.baidu.com/s/1AbgG0yoROHmrRQN7dgzDvQ) (提取码: so6v)|
|NAFNet-GoPro-width64|GoPro|33.7103|0.9668|[link](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing)|[link](https://pan.baidu.com/s/1g-E1x6En-PbYXm94JfI1vg) (提取码: wnwh)|
|NAFNet-SIDD-width32|SIDD|39.9672|0.9599|[link](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing)|[link](https://pan.baidu.com/s/1Xses38SWl-7wuyuhaGNhaw) (提取码: um97)|
|NAFNet-SIDD-width64|SIDD|40.3045|0.9614|[link](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing)|[link](https://pan.baidu.com/s/198kYyVSrY_xZF0jGv9U0sQ) (提取码: dton)|
|NAFNet-REDS-width64|REDS|29.0903|0.8671|[link](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view?usp=sharing)|[link](https://pan.baidu.com/s/1vg89ccbpIxg3mK9IONBfGg) (提取码: 9fas)|

### Image Restoration Tasks

| Task                                 | Dataset | Instructions            | Visualization Results                                        |
| :----------------------------------- | :------ | :---------------------- | :----------------------------------------------------------- |
| Image Deblurring                     | GoPro   | [link](./docs/GoPro.md) | [gdrive](https://drive.google.com/file/d/1S8u4TqQP6eHI81F9yoVR0be-DLh4cNgb/view?usp=sharing)  ｜ [百度云盘](https://pan.baidu.com/s/1yNYQhznChafsbcfHO44aHQ) (提取码: 96ii) |
| Image Denoising                      | SIDD    | [link](./docs/SIDD.md)  | [gdrive](https://drive.google.com/file/d/1rbBYD64bfvbHOrN3HByNg0vz6gHQq7Np/view?usp=sharing)   \|  [百度云盘](https://pan.baidu.com/s/1wIubY6SeXRfZHpp6bAojqQ) (提取码: hu4t) |
| Image Deblurring with JPEG artifacts | REDS    | [link](./docs/REDS.md)  | [gdrive](https://drive.google.com/file/d/1FwHWYPXdPtUkPqckpz-WBitpVyPuXFRi/view?usp=sharing)   \|  [百度云盘](https://pan.baidu.com/s/17T30w5xAtBQQ2P3wawLiVA) (提取码: put5) |



### Citations
If NAFNet helps your research or work, please consider citing NAFNet.

```
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```

### Contact

If you have any questions, please contact chenliangyu@megvii.com or chuxiaojie@megvii.com

---

<details>
<summary>statistics</summary>
 
![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/NAFNet)
 
</details>

