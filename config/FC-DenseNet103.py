from metrics import crossentropy
from lasagne.updates import rmsprop
import imp
import os

# Dataset
dataset = 'camvid'
train_crop_size = (224, 224) # None for full size

# Training
seed = 0
learning_rate = 1e-3
lr_sched_decay = 0.995 # Applied each epocjh
weight_decay = 0.0001
num_epochs = 750
max_patience = 150
loss_function = crossentropy
optimizer = rmsprop # Consider adam for training on other dataset, or decrease epsilon to 1e-12
batch_size = 3

# Architecture
# pretrained_model= None # path of the weights of a pretrained network

model_path = os.path.join(os.getcwd().split('/config')[0], 'FC-DenseNet.py')
net = imp.load_source('Net', model_path).Network(
    input_shape=(None, 3, None, None),
    n_classes=11,
    n_filters_first_conv=48,
    n_pool=5,
    growth_rate=16,
    n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
    dropout_p=0.2)

##############################################################################

if __name__ == '__main__':
    # Display a summary with a given shape for the input image
    net2 = imp.load_source('Net', model_path).Network(
        input_shape=(None, 3, 360, 480),
        n_classes=11,
        n_filters_first_conv=48,
        n_pool=5,
        growth_rate=16,
        n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
        dropout_p=0.2)

    net2.summary()
