## Introduction

This repo contains the code to train and evaluate FC-DenseNets as described in [The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326). We investigate the use of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) for semantic segmentation, and report state of the art results on datasets such as CamVid.

## Installation


You need to install :
- [Theano](https://github.com/Theano/Theano). Preferably the last version
- [Lasagne](https://github.com/Lasagne/Lasagne)
- The dataset loader (**Not yet available**)
- (Recommended) [The new Theano GPU backend](https://github.com/Theano/libgpuarray). Compilation will be much faster.


## Data

The data-loader we used for the experiments will be released later. If you do want to train models now, you need to create a function load_data which returns 3 iterators (for training, validation and test). When applying next(), the iterator returns two values X, Y where X is the batch of input images (shape= (batch_size, 3, n_rows, n_cols), dtype=float32) and Y the batch of target segmentation maps (shape=(batch_size, n_rows, n_cols), dtype=int32) where each pixel in Y is an int indicating the class of the pixel.

The iterator must also have the following methods (so they are not python iterators) : get_n_classes (returns the number of classes), get_n_samples (returns the number of examples in the set), get_n_batches (returns the number of batches necessary to see the entire set) and get_void_labels (returns a list containing the classes associated to void). It might be easier to change directly the files train.py and test.py.


## Run experiments

The architecture of the model is defined in FC-DenseNet.py. To train a model, you need to prepare a configuration file (folder config) where all the parameters needed for creating and training your model are precised. DenseNets contain lot of connections making graph optimization difficult for Theano. We strongly recommend to use the flags described further.

To train the FC-DenseNet103 model, use the command : `THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python train.py -c config/FC-DenseNet103.py -e experiment_name`. All the logs of the experiments are stored in the folder experiment_name.

On a Titan X 12GB, for the model FC-DenseNet103 (see folder config), compilation takes around 400 sec and 1 epoch 120 sec for training and 40 sec for validation.

## Use a pretrained model

We publish the weights of our model FC-DenseNet103. Metrics claimed in the paper (jaccard and accuracy) can be verified running 
`THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python test.py`

## About the "m" number in the paper

There is a small error with the "m" number in the Table 2 of the paper (that you may understand when running the code!). All values from the bottleneck to the last block (880, 1072, 800 and 368) should be incremented by 16 (896, 1088, 816 and 384).

Here how we compute this value representing the number of feature maps concatenated into the "stack" : 
- First convolution : m=48
 - In the downsampling part + bottleneck, m[B] = m[B-1] + n_layers[B] * growth_rate [linear growth]. First block : m = 48 + 4x16 = 112. Second block m = 112 + 5x16 = 192. Until the bottleneck : m = 656 + 15x16 = 896.
- In the upsampling part, m[B] is the sum of 3 terms : the m value corresponding to same resolution in the downsampling part (skip connection), the number of feature maps from the upsampled block (n_layers[B-1] * growth_rate) and the number of feature maps in the new block (n_layers[B] * growth_rate). First upsampling, m =  656 + 15x16 + 12x16 = 1088. Second upsampling, m = 464 + 12x16 + 10x16 = 816. Third upsampling, m = 304 + 10x16 + 7x16 = 576, Fourth upsampling, m = 192 + 7x16 + 5x16 = 384 and fifth upsampling, m = 112 + 5x16 + 4x16 = 256 
    



