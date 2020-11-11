# RS-GAN

This repository contains the code of the following paper:

**Towards a Better Global Loss Landscape of GANs** https://arxiv.org/abs/2011.04926

**Abstract:** Understanding of GAN training is still very limited. One major challenge is its non-convex-non-concave min-max objective, which may lead to sub-optimal local minima. In this work, we perform a global landscape analysis of the empirical loss of GANs. We prove that a class of separable-GAN, including the original JS-GAN, has exponentially many bad basins which are perceived as mode-collapse. We also study the relativistic pairing GAN (RpGAN) loss which couples the generated samples and the true samples. We prove that RpGAN has no bad basins. Experiments on synthetic data show that the predicted bad basin can indeed appear in training. We also perform experiments to support our theory that RpGAN has a better landscape than separable-GAN. For instance, we empirically show that RpGAN performs better than separable-GAN with relatively narrow neural nets.

## Training

There are three main parts of experiments:

* `GAN2_2Cluster.py`: train 1d-cluster experiments for both JS-GAN and RS-GAN. Generated data and projections will be in the folder 1dclusterresults; Evolution of Dloss and generated data will be in the folder DLoss_Data_Evolution.


* `5gaussian.py`: 5 Gaussian experiments. Generated images will be 5gaussian folder.
* `rsgan.py` and `vanillaGAN.py`: Real Image experiments. We provide some commands in scripts.txt



## Evaluation

We provide three evaluation metrics in `eval.py`: FID, Inception Score and Precision and Recall for Distributions.

e.g. to evaluate the regular CNN model's FID on CIFAR-10, do

```
python eval.py --metric FID --dataset cifar --structure dcgan --image_size 32 --num_features 64 --model_path [the/model/path]
```



## Pre-trained Models

We include pretrained models in the folder `pretrained-model`.



## Bibtex

Please cite our work if you find it useful for your research and work:

@ARTICLE{sun2020ganlandscape,
       author = {{Sun}, Ruoyu and {Fang}, Tiantian and {Schwing}, Alex},
        title = "{Towards a Better Global Loss Landscape of GANs}",
        booktitle = {Conference on Neural Information Processing Systems},
        year = 2020
}