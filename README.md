# Differentiable Euler Characteristic Tansform for Molecule prediction

## About

This project is an application of the Differentiable ECT applied to predict molecular data. The theory and motivations behind this work are discussed in my master thesis.

My email is giach96@gmail.com.

### Dataset
The dataset used by the model can be downloaded from [github.com/Boehringer-Ingelheim/topolearn](https://github.com/Boehringer-Ingelheim/topolearn/blob/master/data/raw.tar.gz). It is constituted of 12 different datasets, each one describing a different task.

Once downloaded and unpacked the dataset, by setting `run_type = RunType.PARSE_DATASET` in file `DmpConfig.py` and running file `main.py` the raw data is parsed as `pytorch_geometric` graphs and store in file system.

### Train
The training can be made both by setting `run_type = RunType.TRAIN_MULTIPLE`, which starts a cycle of training for each task, or `run_type = RunType.TRAIN_SINGLE`, which starts a training for the `dataset` specified in the configuration file.

### Target distribution analysis
With `run_type = RunType.ANALYZE_TARGET_DISTRIBUTION` it is possible to visualize the distribution of the target values in the three splits made on the dataset (70% training, 15% test, 15% validation).

### Learned direction analysis
With `run_type = RunType.ANALYZE_DIRECTIONS` one can visualize a 3D plot showing the directions of the ECT of a trained model.

### Acknowledgments

The datasets and the data extraction code come from the [topolearn](https://github.com/Boehringer-Ingelheim/topolearn) repository.

The credits to the DECT layer gives to its inventors, and was published in the [dect](https://github.com/aidos-lab/dect) repository.

Thanks to Professor [Bastian Rieck](https://bastian.rieck.me) for the valuable feedback.