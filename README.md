# Analogy-Based Zero-Shot Learning

This repository contains the code to replicate the experiments in a forthcoming paper on analogy-based zero-shot learning, as well as an improved implementation for the models built upon.

## Code

This repository builds up on existing zero-shot learning models by conditioning the generation of new features on the features of similar seen classes. The intuition is that more information can be gained about a class from the features of similar classes, rather than from the class attributes alone.

The models used in this repository are:

- CLSWGAN: [paper](https://arxiv.org/abs/1712.00981), [official code](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/).
- TFVAEGAN: [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670477.pdf), [official code](https://github.com/akshitac8/tfvaegan).
- FREE: [paper](https://arxiv.org/abs/2107.13807), [official code](https://github.com/shiming-chen/FREE).

## Installation

The code has been tested with Python 3.8.10 and PyTorch 1.13.1+cu116. To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

Before running the code, you need to download the datasets in the `data` folder. 4 datasets are currently supported: AWA2, CUB, SUN, and FLO. The datasets are available [here](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) (if the link doesn't work, copy it in a new tab instead of clicking on it). For each dataset, you only require 2 of the downloaded files: `res101.mat` with the features, and `att_splits.mat` with the attributes. Make sure you have these files in the appropriate folder for each dataset, e.g. `data/awa/res101.mat` and `data/awa/att_splits.mat` for the AWA2 dataset. You can run any of the models, e.g.:

```bash
python run_clswgan.py --dataset CUB --n_epochs 20
```

This will train and evaluate the model on the selected dataset.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
