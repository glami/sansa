# SANSA: how to compute EASE on million item datasets

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-approximate-nonsymmetric-autoencoder/recommendation-systems-on-amazon-book)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-book?p=scalable-approximate-nonsymmetric-autoencoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-approximate-nonsymmetric-autoencoder/collaborative-filtering-on-million-song)](https://paperswithcode.com/sota/collaborative-filtering-on-million-song?p=scalable-approximate-nonsymmetric-autoencoder)

Official implementation of scalable collaborative filtering model **SANSA**.

![Architecture and training procedure of SANSA](assets/sansa.png)

> **Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering**  
> Spišák M., Bartyzal R., Hoskovec A., Peška L., Tůma M.  
> Paper: [10.1145/3604915.3608827](https://doi.org/10.1145/3604915.3608827)
> 
> *Best Short Paper Runner-Up*, [17th ACM Conference on Recommender Systems (ACM RecSys 2023)](https://recsys.acm.org/recsys23/)

### Reproducibility
See branch [reproduce_our_results](https://github.com/glami/sansa/tree/reproduce_our_results) for codes used in experiments and complete experimental results. 

## About

SANSA is a scalable modification of [EASE](https://arxiv.org/abs/1905.03375), a shallow autoencoder for collaborative filtering, **specifically designed to handle item sets with millions of items**.
- End-to-end sparse training procedure: instead of strenuously inverting the Gramian $X^TX$ of user-item interaction matrix $X$, SANSA efficiently finds a *sparse approximate inverse* of $X^TX$. 
- Training memory requirements are proportional to the number of non-zero elements in $X^TX$ (and this can be improved further).  
- The model's density is prescribed via a hyperparameter. 
- As a sparse neural network, SANSA offers *very fast inference* times.

### Learn more in our [short paper](https://dl.acm.org/doi/10.1145/3604915.3608827), or check out the conference [poster](assets/poster.pdf).

## Installation
```bash
pip install sansa
```
(make sure to install prerequisites first, see next section)
### Prerequisites
Training of SANSA uses [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), which depends on the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) numerical library. To install SuiteSparse on Ubuntu and macOS, run the commands below: 
```bash
# Ubuntu
sudo apt-get install libsuitesparse-dev

# macOS
brew install suite-sparse
```
Note that `brew` (and possibly other package managers) installs SuiteSparse objects to non-standard location. Before installing the package, you need to set
the correct path to SuiteSparse by setting the following 2 environment variables:
```bash
export SUITESPARSE_INCLUDE_DIR={PATH TO YOUR SUITESPARSE}/include/suitesparse
export SUITESPARSE_LIBRARY_DIR={PATH TO YOUR SUITESPARSE}/lib
```
For `brew`, you can find `{PATH TO YOUR SUITESPARSE}` by running `brew info suite-sparse`. To streamline this process, you can run
```bash
SUITESPARSE_DIR=$(brew info suitesparse | sed -n 4p | awk '{print $1}')  # path to brew-installed package is on the 4th line, 1st column
export SUITESPARSE_INCLUDE_DIR=$SUITESPARSE_DIR/include/suitesparse
export SUITESPARSE_LIBRARY_DIR=$SUITESPARSE_DIR/lib
```
which should set the correct environment variables for you.

### Installation from source
With SuiteSparse path correctly specified, simply run
```bash
pip install .
```
in the root directory of this repository.

## Usage
### Configuration
SANSA model supports two methods of factorization of the Gramian matrix $X^TX$ and one method for inverting the lower triangular factor. 
Factorizers and inverters are configured separately and included in the model configuration:
```python
from sansa import SANSAConfig

config = SANSAConfig(
    l2 = 20.0,  # regularization strength
    weight_matrix_density = 5e-5,  # desired density of weights
    gramian_factorizer_config = factorizer_config,  # factorizer configuration
    lower_triangle_inverter_config = inverter_config,  # inverter configuration
)
```
To get the configuration of a model instance, use the `config` property:
```python
config = model.config
```
#### Factorizer configuration
Choose between two factorization techniques:
1. **CHOLMOD** = exact Cholesky factorization sparsified after factorization. More accurate but memory-hungry; recommended for smaller, denser matrices.
```python
from sansa import CHOLMODGramianFactorizerConfig

factorizer_config = CHOLMODGramianFactorizerConfig()  # no hyperparameters
```
2. **ICF** = Incomplete Cholesky factorization. Less accurate but much more memory-efficient; recommended for very large, sparse matrices.
```python
from sansa import ICFGramianFactorizerConfig

factorizer_config = ICFGramianFactorizerConfig(
    factorization_shift_step = 1e-3,  # initial diagonal shift if incomplete factorization fails
    factorization_shift_multiplier = 2.0,  # multiplier for the shift for subsequent attempts
)
```
#### Inverter configuration
Currently only one inverter is available: **UMR** -- residual minimization approach
```python
from sansa import UMRUnitLowerTriangleInverterConfig

inverter_config = UMRUnitLowerTriangleInverterConfig(
    scans=1,  # number of scans through all columns of the matrix
    finetune_steps=5,  # number of finetuning steps, targeting worst columns
)
```
### Training
```python
from sansa import SANSA

X = ...  # training data -- scipy.sparse.csr_matrix (rows=users, columns=items)
config = ...  # specify configuration of SANSA model

# Instantiate model with the config
model = SANSA(config)

# Train model on the user-item matrix
model.fit(X)
# or on a precomputed symmetric item-item matrix
model.fit(X, compute_gramian=False)
```
Weights of a SANSA model can be accessed using the `weights` attribute:
```python
w1, w2 = model.weights  # tuple of scipy.sparse.csr_matrix of shape (num_items, num_items)
```
Weights can be loaded into a model using the `load_weights` method:
```python
w1, w2 = ...  # tuple of scipy.sparse.csr_matrix of shape (num_items, num_items)

model.load_weights((w1, w2))
```
### Inference
#### 1. High-level inference: recommendation for a batch of users
```python
X = ...  # input interactions -- scipy.sparse.csr_matrix (rows=users, columns=items)

# Get indices of top-k items for each user + corresponding scores
# if mask_input=True, input items get score=0
top_k_indices, top_k_scores = model.recommend(X, k=10, mask_input=True)  # np.ndarrays of shape (X.shape[0], k)
```
#### 2. Low-level inference: forward pass
```python
X = ...  # input interactions -- scipy.sparse.csr_matrix (rows=users, columns=items)

# Forward pass
scores = model.forward(X)  # scipy.sparse.csr_matrix of shape X.shape
```

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
abstract = {In the field of recommender systems, shallow autoencoders have recently gained significant attention. One of the most highly acclaimed shallow autoencoders is easer, favored for its competitive recommendation accuracy and simultaneous simplicity. However, the poor scalability of easer (both in time and especially in memory) severely restricts its use in production environments with vast item sets. In this paper, we propose a hyperefficient factorization technique for sparse approximate inversion of the data-Gram matrix used in easer. The resulting autoencoder, sansa, is an end-to-end sparse solution with prescribable density and almost arbitrarily low memory requirements — even for training. As such, sansa allows us to effortlessly scale the concept of easer to millions of items and beyond.},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {763–770},
numpages = {8},
keywords = {Algorithm scalability, Numerical approximation, Sparse approximate inverse, Sparse autoencoders},
location = {Singapore, Singapore},
series = {RecSys '23}
}
```
