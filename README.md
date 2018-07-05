# Deploy-time Network Optimization Tool Box

In this repository, we released code for the combination of different deploy time network optimization algorithm.
- spatial decomposition (SD)
- network decoupling (ND)
- channel decomposition (CD)

The current network supported: VGG-16, ResNet series, DenseNet series and AlexNet-bn.
    
### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#channel-pruning) 
4. [Experiment Result](#experiment-results) 
4. [Reference](#reference)

### Requirements
1. Python3 packages you might not have: `scipy`, `sklearn`, `easydict`, use `sudo pip3 install` to install.
2. An NVIDIA GPU is recommanded.

### Installation
1. Clone the repository of Caffe and compile it
```Shell
    git clone https://github.com/BVLC/caffe.git
    cd caffe
    # modify Makefile.config to the path of the library on your machine, please make sure the python3 interface is supported
    make -j8
    make pycaffe
```
2. Clone this repository 
```Shell
    https://github.com/lyxok1/Deploy-Time-CNN-Optimization.git
```
    
### Usage  
1. Download the original model files (.prototxt and .caffemodel) and move them to the directory of `models`

2. Make proper configurations in `config.py`
   To make sure the network optimization works well, please enter the file `config.py` and change the configuration of the parameters according to the comment above them.

   Note, among the hyperparameters above the `SD_Param`,`ND_Param`,`CD_Param` and `device_id` could also be specified in command line (see section 3), while other parameters must be set correctly according to the comment above.

3. Command Line Usage
To decouple a network, use the following command
```Shell
    python3.5 main.py <optional arguments>
    optional arguments:
        -h, --help            show this help message and exit
        -sd                   enable spatial decomposition
        -nd                   enable network decoupling
        -cd                   enable channel decompostion
        -data                 enable data driven for spatial decomposition
        -speed SPEED          sd speed up ratio of spatial decomposition
        -threshold THRESHOLD  energy threshold for network decouple
        -gpu GPU              caffe devices
        -model MODEL          caffe prototxt file path
        -weight WEIGHT        caffemodel file path
        -action {decomp,compute,test}
                              compute, test or decompose the model
        -iterations ITER      test iterations
        -rank RANK            rank for network decoupling
        -DP                   flags to set DW + PW decouple (default is PW + DW)

```

For example, suppose the VGG-16 network is in folder `models/` and named as `vgg.prototxt` and `vgg.caffemodel`, you could use the following command to conduct network decoupling (ND) with threshold 0.95:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -threshold 0.95
```
Or you can decouple the network using a given rank instead of threshold:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -rank 5
```
Similarly, you could decompose the model with spatial decomposition with compression ratio of 2 and data reconstruction:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -sd -speed 2.0 -data
```

Note: the decomposition results are saved under the directory of `models`, with the name format of `new_xx_relu_separ_xx.prototxt`, where the first `xx` is the combination string of compression ratio and/or ND threshold, and the second `xx` is the original model name. The result weights file is also saved under `models` with the name format of `new_decomp_merged_xx.caffemodel` where `xx` is the original model name of input. 

When the `action` FLAG is `compute`, the program will compute the FLOPs of specified model. When the `action` FLAG is `test`, the program will test the model accuracy on specified dataset.

`model`,`weight` and `action` must be specified in command line. As for other arguments, if they are not specified in command line, the system will use the default value in configuration file `config.py` (see section 2).

The combinations of sd + cd and sd + nd + cd are not supported now.

### Experiment Result
Here are some results from experiments conducted on VGG-16 network, the original FLOPs is 15.35G

| Method | Decomposed FLOPs | Accuracy drop |
| ------ | ------|------|
|ND| 8.61G | 1% |
|CD| 6.52G | 1% |
|SD(no data)| 7.20G | 1% |
|SD(data)| 4.28G | 1% |

### Reference

This work is based on our work *Network Decoupling: From Regular Convolution to Separable Depthwise Convolution (BMVC2018)*. If you think this is helpful for your research, please consider append following bibtex config in your latex file.

```Latex
@inproceedings{guo2018nd,
  title = {{Network Decoupling}: From Regular Convolution to Separable Depthwise Convolution},
  author = {Guo, Jianbo and Li, Yuxi and Li, Jianguo and Lin, Weiyao},
  booktitle = {BMVC},
  year = {2018}
}
```

This repository is also referenced to the work of channel decomposition and spatial decomposition, to know more details about channel decomposition and spatial decomposition, please refer to following papers.
- [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
- [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)
