# Scalable Approximate NonSymmetric Autoencoder (SANSA)

### Abstract
In the field of recommender systems, shallow autoencoders have recently gained significant attention. One of the most highly acclaimed shallow autoencoders is EASE, favored for its competitive recommendation accuracy and simultaneous simplicity. However, the poor scalability of EASE (both in time and especially in memory) severely restricts its use in production environments with vast item sets.

In this paper, we propose a hyperefficient factorization technique for sparse approximate inversion of the data-Gram matrix used in EASE. The resulting autoencoder, SANSA, is an end-to-end sparse solution with prescribable density and almost arbitrarily low memory requirements (even for training). As such, SANSA allows us to effortlessly scale the concept of EASE to millions of items and beyond.

### Model
![Architecture and training procedure of SANSA](https://gcdnb.pbrd.co/images/UrdzeLSiVDYg.png?raw=True "Architecture and training procedure of SANSA")

### Data
5 datasets are available for experiments:
1. `goodbooks10`: Goodbooks-10k dataset ([link](https://github.com/zygmuntz/goodbooks-10k)).
2. `movielens20`: MovieLens 20M dataset ([link](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)).
3. `msd`: Million Song Dataset ([link](https://www.kaggle.com/competitions/msdchallenge/data)).
4. `netflix`: Netflix Prize dataset ([link](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)).
5. `amazonbook`: Amazon Books dataset ([link](https://github.com/kuandeng/LightGCN/tree/master/Data/amazon-book))

The dataset files should be located inside `datasets/data/{dataset_name}`.

### Setting up virtual environment using Conda
Updating the conda installation first is recommended: `conda update -n base -c conda-forge conda`

#### Recommended
Intel optimized (but also works on AMD).
```
conda create -n sansa python==3.10.9
conda activate sansa

conda install -c intel numpy==1.22.3 scipy==1.7.3
conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8

pip install sparse-dot-mkl==0.8.3 black==23.3.0 numba==0.57.0 memory-profiler==0.61.0 pandas==2.0.1 scikit-learn==1.2.2 fastparquet==2023.4.0 matplotlib==3.7.1 jupyter==1.0.0
```

#### Compatibility mode
Works on Apple Silicon. Works when MKL is not available.
```
conda create -n sansa-nomkl python==3.10.9
conda activate sansa-nomkl

conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8

pip install numpy==1.22.3 scipy==1.7.3 black==23.3.0 numba==0.57.0 memory-profiler==0.61.0 pandas==2.0.1 scikit-learn==1.2.2 fastparquet==2023.4.0 matplotlib==3.7.1 jupyter==1.0.0
```

## Reproducing the results
1. Download the datasets and store them in the `datasets/data` folder.
2. Setup a virtual environment using the instructions above.
3. Run experiments from the **root** directory:
```
python experiments/{experiment_name}/{dataset_name}/run_experiment{_optional_specifiers}.py
```
The results (json) are located in `experiments/{experiment_name}/{dataset_name}/results`.
