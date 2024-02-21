# unsupervised-semantic-analysis

Code developed for the research work: 

C. Echegoyen, A. Pérez, G. Santafé, U. Pérez-Goya and M.D Ugarte. [Large-scale unsupervised spatio-temporal semantic analysis of vast regions from satellite images sequences](https://link.springer.com/article/10.1007/s11222-024-10383-y?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240205&utm_content=10.1007/s11222-024-10383-y). Statistics and Computing 34, 71 (2024).

We provide python notebooks with experiments of the paper in the directory `experiments`.  

An example of sequence with 3 Sentinel-2 images and the embedding models trained for the paper can be dowolad [here](https://emi-sstcdapp.unavarra.es/unsupervised-semantic-analysis.zip). The models should be in the directory `models` and the sequence of satellite images in the directory `data/NE-TXN`. The rest of images are not provided here due to space constraints, but they can be shared upon request.

The directory `src` contains the [Tile2Vec code](https://github.com/ermongroup/tile2vec) with some mofications and additional functions. As indicated by the authors, the Tile2Vec LICENSE is included in this repository.

The following notebooks must be run in this order:
1. Create_grid_MTS: loads a model and creates a collection of multivariate time series by embedding sequnces of tiles.
2. Experiment_1: runs and plots the clustering of time series in differente ways, calculates qualitiy measures and explores the semantic provided by the clustering

Note that the results of these illustrative examples correspond to a sequence of 3 images. In the paper, we analyze a region covered by 4 images and use sequences of 20 images.
