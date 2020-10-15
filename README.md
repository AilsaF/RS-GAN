# RS-GAN
This repository contains the code of the following paper:
> **On Understanding the Global Landscape of Generative Adversarial Nets**<br>
>
> **Abstract:** *Understanding of GAN training is still very limited. One major challenge is the non-convex-non-concave min-max objective in GAN training, which may lead to sub-optimal local minima. In this work, we perform a global landscape analysis of GANs using the empirical version. We prove that the original GAN formulation has exponentially many bad basins which are perceived as mode-collapse. We also study a modification to the original loss called RS-GAN (relative standard GAN) which couples the generated samples and the true samples, and prove that RS-GAN has no bad basins. Experiments on the synthetic data show that the predicted bad basin can indeed appear in training. We also perform a few experiments to support our theory that RS-GAN has a better landscape than JS-GAN. For instance, we empirically show that RS-GAN performs better than JS-GAN with relatively narrow neural nets.

## Training
There are three main parts of experiments:
* GAN2_2Cluster.py: train 1d-cluster experiments for both JS-GAN and RS-GAN. Generated data and projections will be in the folder 1dclusterresults; Evolution of Dloss and generated data will be in the folder DLoss_Data_Evolution.


* 5gaussian.py: 5 Gaussian experiments. Generated images will be 5gaussian folder.

* rsgan.py and vanillaGAN.py: Real Image experiments. We provide some commands in scripts.txt
