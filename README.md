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
    git clone https://github.com/Betterthinking/network-decoupling.git
```
    
### Usage  
1. Download the original model files (.prototxt and .caffemodel) and move them to the directory of `models`

2. Make proper configurations in `config.py`
   To make sure the network optimization works well, please enter the file `config.py` and change the configuration of the parameters according to the comment above them. Here is an example of `config.py`
```Python
   from easydict import EasyDict as edict
    # construct parameters for different method
    SD_param = edict()
    ND_param = edict()
    CD_param = edict()
    # decide if SD is driven by data
    SD_param['data_driven'] = False
    # FLOPs compression ratio (per layer) for each layer in SD
    SD_param['c_ratio'] = 3
    # enable trigger for SD
    SD_param['enable'] = False
    # threshold of energy ratio for network decoupling
    ND_param['energy_threshold'] = 0.9
    # enable trigger for ND
    ND_param['enable'] = True
    # FLOPs compression ratio (per layer) for each layer in CD
    CD_param['c_ratio'] = 2
    # enable trigger for CD
    CD_param['enable'] = False 
    # dict containg layers not requiring decomposition
    mask_layers = ['conv1','fc6']
    # gpu device (-1 for CPU)
    device_id = 0

    # parameters for data driven decoupling
    # the input layer name of network
    data_layer = 'data'
    # the dataset used for data reconstruction
    dataset = 'imagenet'
    # samples of batches for data reconstruction
    nSamples = 600
    # extract how many points per sample
    nPointsPerSample = 10
    # accurate or mAP layer names for data driven method (default value is accuracy@5 in vgg-16)
    accname = 'acc/top-5'
    # the name of frozen pickle to store sample points
    frozen_name = 'frozen'
    # test param, the path to your caffe binary file
    caffe_path = '/home/jli59/yuxili/ker2col-caffe/build/tools/caffe'
    # imagenet val source
    imagenet_val = '/data/sde/jli59/jianbo/lmdb/ilsvrc12_val_lmdb'
    # cifar10 val source
    cifar10_val = '/path/to/cifar10_val'
```

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
```

For example, suppose the VGG-16 network is in folder `models/` and named as `vgg.prototxt` and `vgg.caffemodel`, you could use the following command to conduct network decoupling (ND) with threshold 0.95:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -threshold 0.95
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
This repository is build based on the work of [channel-pruning](https://github.com/yihui-he/channel-pruning.git), many thanks to the author of this work ([Yihui He](http://yihui-he.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en&oi=ao) and [Jian Sun](http://jiansun.org/))

To know more details about channel decomposition and spatial decomposition, please refer to following papers.
- [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
- [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)
