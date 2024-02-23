# Unsupervised semantic analysis

This repository provides the Python code to reproduce the computational analysis of the paper: "Large-scale unsupervised spatio-temporal semantic analysis of vast regions from satellite images sequences" (Echegoyen et al., 2024).

## Table of contents

- [Code](#R-code)
- [Data and models](#Additional-data)

## Code

We provide python notebooks with experiments of the paper in the directory `experiments`.  

The directory `src` contains the [Tile2Vec code](https://github.com/ermongroup/tile2vec) with some mofications and additional functions. As indicated by the authors, the Tile2Vec LICENSE is included in this repository.

The following notebooks must be run in this order:
1. Create_grid_MTS: loads a model and creates a collection of multivariate time series by embedding sequnces of tiles.
2. Experiment_1: runs and plots the clustering of time series in differente ways, calculates qualitiy measures and explores the semantic provided by the clustering

## Data and models

An example of sequence with 3 Sentinel-2 images and the embedding models trained for the paper can be dowolad [here](https://emi-sstcdapp.unavarra.es/unsupervised-semantic-analysis.zip). The models should be in the directory `models` and the sequence of satellite images in the directory `data/NE-TXN`. The rest of images are not provided here due to space constraints, but they can be shared upon request.

Note that the results of these illustrative examples correspond to a sequence of 3 images. In the paper, we analyze a region covered by 4 images and use sequences of 20 images.

# Acknowledgements

This work has been supported by Project PID2020-113125RB-I00/MCIN/AEI/10.130 39/501100011033. Aritz Pérez has been supported by Basque Government through the Elkartek program and the BERC 2022-2025 program, and by the Ministry of Science and Innovation: BCAM Severo Ochoa accreditation CEX2021-001142-S/ MICIN/ AEI/ 10.13039/ 501100011033.

![image](https://github.com/spatialstatisticsupna/LXG/blob/main/micin-aei.jpg)

# Reference

C. Echegoyen, A. Pérez, G. Santafé, U. Pérez-Goya and M.D Ugarte. [Large-scale unsupervised spatio-temporal semantic analysis of vast regions from satellite images sequences](https://doi.org/10.1007/s11222-024-10383-y). Statistics and Computing 34, 71 (2024).
