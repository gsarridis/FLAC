# FLAC: Fairness-Aware Representation Learning by Suppressing Attribute-Class Associations
This is a pytorch implementation of [FLAC](https://arxiv.org/abs/2304.14252), a methodology that minimizes mutual information between the features extracted by the model and a protected attribute, without the use of attribute labels. To do that, FLAC proposes a sampling strategy that highlights underrepresented samples in the dataset, and casts the problem of learning fair representations as a probability matching problem that leverages representations extracted by a bias-capturing classifier. It is theoretically shown that FLAC can indeed lead to fair representations, that are independent of the protected attributes.
## Python environment

python = 3.6
pytorch = 1.10.1

We also provide the .yml file of the environmnent we used (pytorch36_environment.yml).
You can create an identical environment by running
````
conda env create -f pytorch36_environment.yml
````
## Run the code
1.Download and extract UTKFace, CelebA, Imagenet, and Imagenet-A to a data folder.

2.Download the provided [bias-capturing classifiers](https://github.com/gsarridis/FLAC/releases/tag/bcc) and put them under the bias_capturing_classifiers folder.

3.For Biased-MNIST run: 
````
python train_biased_mnist.py --corr 0.99 --alpha 110
python train_biased_mnist.py --corr 0.995 --alpha 1500
python train_biased_mnist.py --corr 0.997 --alpha 2000
python train_biased_mnist.py --corr 0.999 --alpha 10000
````

4.For UTKFace run: 
````
python train_utk_face.py --task race
python train_utk_face.py --task age
````

5.For CelebA run: 
````
python train_celeba.py --task blonde --alpha 30000
python train_celeba.py --task makeup --alpha 20000
````

6.For Imagenet run: 
````
python get_imagenet_bias_features.py
python train_imagenet.py
````

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/classification-r-0-999-on-biased-mnist)](https://paperswithcode.com/sota/classification-r-0-999-on-biased-mnist?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/classification-r-0-997-on-biased-mnist)](https://paperswithcode.com/sota/classification-r-0-997-on-biased-mnist?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/classification-r-0-995-on-biased-mnist)](https://paperswithcode.com/sota/classification-r-0-995-on-biased-mnist?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/classification-r-0-990-on-biased-mnist)](https://paperswithcode.com/sota/classification-r-0-990-on-biased-mnist?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/heavymakeup-bias-conflicting-on-celeba)](https://paperswithcode.com/sota/heavymakeup-bias-conflicting-on-celeba?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/heavymakeup-unbiased-on-celeba)](https://paperswithcode.com/sota/heavymakeup-unbiased-on-celeba?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/haircolor-bias-conflicting-on-celeba)](https://paperswithcode.com/sota/haircolor-bias-conflicting-on-celeba?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/haircolor-unbiased-on-celeba)](https://paperswithcode.com/sota/haircolor-unbiased-on-celeba?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/age-unbiased-on-utkface)](https://paperswithcode.com/sota/age-unbiased-on-utkface?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/age-bias-conflicting-on-utkface)](https://paperswithcode.com/sota/age-bias-conflicting-on-utkface?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/race-unbiased-on-utkface)](https://paperswithcode.com/sota/race-unbiased-on-utkface?p=flac-fairness-aware-representation-learning)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/flac-fairness-aware-representation-learning/race-bias-conflicting-on-utkface)](https://paperswithcode.com/sota/race-bias-conflicting-on-utkface?p=flac-fairness-aware-representation-learning)

## Cite
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
