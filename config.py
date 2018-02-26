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
 

# dict containg layers not requiring spatial decomposition
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
# test param
caffe_path = '/home/jli59/yuxili/ker2col-caffe/build/tools/caffe'
# imagenet val source
imagenet_val = '/data/sde/jli59/jianbo/lmdb/ilsvrc12_val_lmdb'
# cifar10 val source
cifar10_val = '/path/to/cifar10_val'
