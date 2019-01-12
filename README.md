# BigGAN-Tensorflow
Simple Tensorflow implementation of ["Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)](https://arxiv.org/abs/1809.11096)

![main](./assets/main.png)

## Issue
* **The paper** used `orthogonal initialization`, but `I used random normal initialization.` The reason is, when using the orthogonal initialization, it did not train properly.

## Usage
### dataset
* `mnist` and `cifar10` are used inside keras
* For `your dataset`, put images like this:

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
```
### train
* python main.py --phase train --dataset celebA-HQ --gan_type hinge

### test
* python main.py --phase test --dataset celebA-HQ --gan_type hinge

## Architecture
<img src = './assets/architecture.png' width = '600px'> 

### 128x128
<img src = './assets/128.png' width = '600px'> 

### 256x256
<img src = './assets/256.png' width = '600px'> 

### 512x512
<img src = './assets/512.png' width = '600px'> 

## Author
Junho Kim
