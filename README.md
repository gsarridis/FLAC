# FLAC: Fairness-Aware Representation Learning by Suppressing Attribute-Class Associations
[![MAI_BIAS toolkit](https://img.shields.io/badge/MAI_BIAS-fairness_tools-blue?logo=github)](https://mammoth-eu.github.io/mammoth-commons/index.html)

This software is part of MAI-BIAS; a low-code toolkit for
fairness analysis and mitigation, with an accompanying suite of coding
tools. Our ecosystem operates in multidimensional and multi-attribute
settings (safeguarding multiple races, genders, etc), and across multiple
data modalities (like tabular data, images, text, graphs). Learn more
[here](https://mammoth-eu.github.io/mammoth-commons/index.html).

---
## 🌍 Overview
<p align="center">
  <img width="438" height="346" alt="image" src="https://github.com/user-attachments/assets/744e3f66-ef46-4665-abed-1123fa67b144" />
</p>

[**FLAC**](https://ieeexplore.ieee.org/document/10737139) is a state-of-the-art method for learning **fair visual representations** by reducing the influence of hidden biases — **without requiring protected attribute labels**.

It enables teams to build AI systems that are:
- More **robust**
- More **trustworthy**
- Better aligned with **fairness and regulatory requirements**

Unlike traditional approaches, FLAC does not rely on demographic annotations (e.g., gender, race), making it highly applicable in **real-world scenarios where such labels are unavailable or restricted**.

---

## 🎯 Who is this for?
- **AI researchers** working on fairness and representation learning  
- **ML engineers** building production-ready vision systems  
- **Decision-makers & product owners** aiming to deploy fair and compliant AI  

---

## 💡 Why FLAC matters (even for non-experts)
- ✅ **No sensitive labels required** → avoids privacy and legal risks  
- ✅ **Fairness by design** → reduces bias during training  
- ✅ **Maintains strong performance** while improving fairness  
- ✅ **Works in real-world conditions** with incomplete data  

---

## 🔥 Key Features
- Minimizes dependence between learned features and hidden biases  
- Does **not require protected attribute annotations**  
- Uses a **sampling strategy** to highlight underrepresented data  
- Leverages a **bias-capturing classifier** to guide training  
- Theoretically grounded with guarantees for fair representations  

---

## ⚡ Python Environment
```
python = 3.6  
pytorch = 1.10.1  
```
We also provide the environment file used in our experiments:
```
conda env create -f pytorch36_environment.yml
```
---

## ⚡ Run the Code

### 1. Prepare datasets
Download and extract:
- UTKFace  
- CelebA  
- ImageNet  
- ImageNet-A  

Place them in a data directory.

---

### 2. Download bias-capturing classifiers
Download from:  https://github.com/gsarridis/FLAC/releases/tag/bcc  

Place them under: bias_capturing_classifiers/

---

### 3. Run experiments

#### Biased-MNIST
```
python train_biased_mnist.py --corr 0.99 --alpha 110  
python train_biased_mnist.py --corr 0.995 --alpha 1500  
python train_biased_mnist.py --corr 0.997 --alpha 2000  
python train_biased_mnist.py --corr 0.999 --alpha 10000  
```
---

#### UTKFace
```
python train_utk_face.py --task race  
python train_utk_face.py --task age  
```
---

#### CelebA
```
python train_celeba.py --task blonde --alpha 30000  
python train_celeba.py --task makeup --alpha 20000  
```
---

#### ImageNet
```
python get_imagenet_bias_features.py  
python train_imagenet.py  
```
---

## 📖 Citation
If you find this code useful in your research, please consider citing:
```
@article{sarridis2024flac,  
  title={Flac: Fairness-aware representation learning by suppressing attribute-class associations},  
  author={Sarridis, Ioannis and Koutlis, Christos and Papadopoulos, Symeon and Diou, Christos},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  year={2024},  
  publisher={IEEE}  
}
```
