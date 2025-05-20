# *Color:* Layered Score Distillation for Recoloring

## Setup
### Install dependencies
````
conda create -y -n lasagna python=3.10.11
conda activate lasagna
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install accelerate
pip install git+https://github.com/huggingface/diffusers

````

## Training Lasagna to relight an input image
```
bash train_color_digital_art.sh
```
