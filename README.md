# Bridging the Gap Between Real and Synthetic Traffic Sign Repositories

Diogo Lopes da Silva and António Ramires Fernandes

to be published in proceedings of Delta 2022

This work aims at generating synthetic traffic sign datasets. These datasets can be be used to train CNN models that provide high levels of accuracy when tested with real data.

We employ traditional techniques for the generation of our synthetic samples, and explore two new operators: Perlin and Confetti Noise. These two operators proved essential in helping us achieve accuracies that are extremelly close to those obtained with real data datasets.

These datasets are created based only on a set of templates. We tested our synthetic datasets against three known Traffic Sign datasets:

- [GTSRB](https://benchmark.ini.rub.de/)
- BTSC
- rMASTIF

Here are some samples generated by our script for the German Traffic Sign Repository Benchmark:

![German synthetic samples](/images/gtsrb_synth.jpg)

When real data is available it is possible to add more information to the generator. In our work we explored adding brightness information. We found that all the studied datasets obbeyed a Johnson distribution regarding brightness and were able to obtain the parameters of such distribution for each data set. Based on this information we were able to generate samples with brightness values from the respective distribution.

In the absence of real data, the overall brightness of the image is computed as:

$B = bias + u^\gamma \times (255.bias)$ (Eq. 1)

where $bias$ determines the minimum brightness, and $u$ is a sample from an uniform distribution between[0,1]. In  our tests we set $bias=10$ and $\gamma = 2$

Most works so far use real scenery images as backgrounds for the synthetic samples. We also explored applying solid colour backgrounds. 

All options included we were able to generate four different datasets:

- SES: Synthetic dataset with brightness drawn from exponential equation (Eq. 1) and solid color backgrounds.
- SER: Synthetic dataset with brightness drawn from exponential equation (Eq. 1) and real image backgrounds.
- SJS: Synthetic dataset with brightness drawn from Johnson distribution and solid color backgrounds.
- SJR: Synthetic dataset with brightness drawn from Johnson distribution and real image backgrounds.

As opposed to previous works such as [1] and [2] we didn´t aim at achieving photo-realistic imagery for our synthetic samples, yet we were able to achieve state of the art results with our approach. 

### Refs

[1] Luo, H., Kong, Q., and Wu, F. (2018). Traffic sign image synthesis with generative adversarial networks.
In 2018 24th International Conference on Pattern Recognition (ICPR), pages 2540–2545.

[2] Spata, D., Horn, D., and Houben, S. (2019). Generation
of natural traffic sign images using domain translation with cycle-consistent generative adversarial networks. In 2019 IEEE Intelligent Vehicles Symposium
(IV), pages 702–708.

