<!--
Copyright 2023 Inspigroup s.r.o.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://github.com/glami/sansa/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
-->
# SANSA: how to compute EASE on million item datasets

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-approximate-nonsymmetric-autoencoder/recommendation-systems-on-amazon-book)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-book?p=scalable-approximate-nonsymmetric-autoencoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-approximate-nonsymmetric-autoencoder/collaborative-filtering-on-million-song)](https://paperswithcode.com/sota/collaborative-filtering-on-million-song?p=scalable-approximate-nonsymmetric-autoencoder)

Official implementation of scalable collaborative filtering model SANSA (*Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering*, Spišák et al., 17th ACM Conference on Recommender Systems (ACM RecSys 2023), Singapore, Best Short Paper Runner-Up). 

### Try it yourself in [Google Colab](https://colab.research.google.com/drive/1ktQMuROR7Z528JKJDQh3GdTatDmWbDHQ?usp=sharing).

### Read our short paper [here](https://dl.acm.org/doi/10.1145/3604915.3608827), or check out the conference [poster](https://github.com/glami/sansa/blob/main/poster.pdf).

### Reproducibility
See branch [reproduce_our_results](https://github.com/glami/sansa/tree/reproduce_our_results) for codes used in experiments and complete experimental results. 

### Abstract
In the field of recommender systems, shallow autoencoders have recently gained significant attention. One of the most highly acclaimed shallow autoencoders is EASE, favored for its competitive recommendation accuracy and simultaneous simplicity. However, the poor scalability of EASE (both in time and especially in memory) severely restricts its use in production environments with vast item sets.

In this paper, we propose a hyperefficient factorization technique for sparse approximate inversion of the data-Gram matrix used in EASE. The resulting autoencoder, SANSA, is an end-to-end sparse solution with prescribable density and almost arbitrarily low memory requirements (even for training). As such, SANSA allows us to effortlessly scale the concept of EASE to millions of items and beyond.

## Model
![Architecture and training procedure of SANSA](https://github.com/glami/sansa/blob/main/sansa.png)

## Setting up virtual environment using Conda
Updating the conda installation first is recommended: `conda update -n base -c conda-forge conda`

### Recommended
Intel optimized (but also works on AMD).
```
conda create -n sansa python==3.10.9
conda activate sansa

conda install -c intel numpy==1.22.3 scipy==1.7.3
conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8

pip install sparse-dot-mkl==0.8.3 black==23.3.0 numba==0.57.0 memory-profiler==0.61.0 pandas==2.0.1 scikit-learn==1.2.2 fastparquet==2023.4.0 matplotlib==3.7.1 jupyter==1.0.0
```

### Compatibility mode
Works on Apple Silicon. Works when MKL is not available.
```
conda create -n sansa-nomkl python==3.10.9
conda activate sansa-nomkl

conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8

pip install numpy==1.22.3 scipy==1.7.3 black==23.3.0 numba==0.57.0 memory-profiler==0.61.0 pandas==2.0.1 scikit-learn==1.2.2 fastparquet==2023.4.0 matplotlib==3.7.1 jupyter==1.0.0
```

### Data
5 datasets are available for experiments:
1. `goodbooks10`: Goodbooks-10k dataset ([link](https://github.com/zygmuntz/goodbooks-10k)).
2. `movielens20`: MovieLens 20M dataset ([link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)).
3. `msd`: Million Song Dataset ([link](https://www.kaggle.com/competitions/msdchallenge/data)).
4. `netflix`: Netflix Prize dataset ([link](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)).
5. `amazonbook`: Amazon Books dataset ([link](https://github.com/kuandeng/LightGCN/tree/master/Data/amazon-book)).

The dataset files should be located inside `datasets/data/{dataset_name}`.

## License
Copyright 2023 Inspigroup s.r.o.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[https://github.com/glami/sansa/blob/main/LICENSE](https://github.com/glami/sansa/blob/main/LICENSE)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Cite us
Please consider citing our paper:
```
@inproceedings{10.1145/3604915.3608827,
author = {Spi\v{s}\'{a}k, Martin and Bartyzal, Radek and Hoskovec, Anton\'{\i}n and Peska, Ladislav and T\r{u}ma, Miroslav},
title = {Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering},
year = {2023},
isbn = {9798400702419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3608827},
doi = {10.1145/3604915.3608827},
abstract = {In the&nbsp;field of recommender systems, shallow autoencoders have recently gained significant attention. One of the&nbsp;most highly acclaimed shallow autoencoders is easer, favored for its competitive recommendation accuracy and simultaneous simplicity. However, the&nbsp;poor scalability of easer (both in time and especially in memory) severely restricts its use in production environments with vast item sets. In this paper, we propose a&nbsp;hyperefficient factorization technique for sparse approximate inversion of the&nbsp;data-Gram matrix used in easer. The&nbsp;resulting autoencoder, sansa, is an&nbsp;end-to-end sparse solution with prescribable density and almost arbitrarily low memory requirements — even for training. As such, sansa allows us to effortlessly scale the&nbsp;concept of easer to millions of items and beyond.},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {763–770},
numpages = {8},
keywords = {Numerical approximation, Algorithm scalability, Sparse autoencoders, Sparse approximate inverse},
location = {Singapore, Singapore},
series = {RecSys '23}
}
```
