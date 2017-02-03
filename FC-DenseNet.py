import numpy as np
import theano.tensor as T
from lasagne.init import HeUniform
from lasagne.layers import (InputLayer, ConcatLayer, Conv2DLayer, Pool2DLayer, Deconv2DLayer,
                            get_all_param_values, set_all_param_values, get_output_shape, get_all_layers)
from lasagne.nonlinearities import linear

from layers import BN_ReLU_Conv, TransitionDown, TransitionUp, SoftmaxLayer


class Network():
    def __init__(self,
                 input_shape=(None, 3, None, None),
                 n_classes=11,
                 n_filters_first_conv=48,
                 n_pool=4,
                 growth_rate=12,
                 n_layers_per_block=5,
                 dropout_p=0.2):
        """
        This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
        The network consist of a downsampling path, where dense blocks and transition down are applied, followed
        by an upsampling path where transition up and dense blocks are applied.
        Skip connections are used between the downsampling path and the upsampling path
        Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.

        :param input_shape: shape of the input batch. Only the first dimension (n_channels) is needed
        :param n_classes: number of classes
        :param n_filters_first_conv: number of filters for the first convolution applied
        :param n_pool: number of pooling layers = number of transition down = number of transition up
        :param growth_rate: number of new feature maps created by each layer in a dense block
        :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
        :param dropout_p: dropout rate applied after each convolution (0. for not using)
        """

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError

        # Theano variables
        self.input_var = T.tensor4('input_var', dtype='float32')  # input image
        self.target_var = T.tensor4('target_var', dtype='int32')  # target

        #####################
        # First Convolution #
        #####################

        inputs = InputLayer(input_shape, self.input_var)

        # We perform a first convolution. All the features maps will be stored in the tensor called stack (the Tiramisu)
        stack = Conv2DLayer(inputs, n_filters_first_conv, filter_size=3, pad='same', W=HeUniform(gain='relu'),
                            nonlinearity=linear, flip_filters=False)
        # The number of feature maps in the stack is stored in the variable n_filters
        n_filters = n_filters_first_conv

        #####################
        # Downsampling path #
        #####################

        skip_connection_list = []

        for i in range(n_pool):
            # Dense Block
            for j in range(n_layers_per_block[i]):
                # Compute new feature maps
                l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
                # And stack it : the Tiramisu is growing
                stack = ConcatLayer([stack, l])
                n_filters += growth_rate
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = TransitionDown(stack, n_filters, dropout_p)

        skip_connection_list = skip_connection_list[::-1]

        #####################
        #     Bottleneck    #
        #####################

        # We store now the output of the next dense block in a list. We will only upsample these new feature maps
        block_to_upsample = []

        # Dense Block
        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = ConcatLayer([stack, l])

        #######################
        #   Upsampling path   #
        #######################

        for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

            # Dense Block
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
                block_to_upsample.append(l)
                stack = ConcatLayer([stack, l])

        #####################
        #      Softmax      #
        #####################

        self.output_layer = SoftmaxLayer(stack, n_classes)

    ################################################################################################################

    def save(self, path):
        """ Save the weights """
        np.savez(path, *get_all_param_values(self.output_layer))

    def restore(self, path):
        """ Load the weights """

        with np.load(path) as f:
            saved_params_values = [f['arr_%d' % i] for i in range(len(f.files))]
        set_all_param_values(self.output_layer, saved_params_values)

    def summary(self, light=False):
        """ Print a summary of the network architecture """

        layer_list = get_all_layers(self.output_layer)

        def filter_function(layer):
            """ We only display the layers in the list below"""
            return np.any([isinstance(layer, layer_type) for layer_type in
                           [InputLayer, Conv2DLayer, Pool2DLayer, Deconv2DLayer, ConcatLayer]])

        layer_list = filter(filter_function, layer_list)
        output_shape_list = map(get_output_shape, layer_list)
        layer_name_function = lambda s: str(s).split('.')[3].split('Layer')[0]

        if not light:
            print('-' * 75)
            print 'Warning : all the layers are not displayed \n'
            print '    {:<15} {:<20} {:<20}'.format('Layer', 'Output shape', 'W shape')

            for i, (layer, output_shape) in enumerate(zip(layer_list, output_shape_list)):
                if hasattr(layer, 'W'):
                    input_shape = layer.W.get_value().shape
                else:
                    input_shape = ''

                print '{:<3} {:<15} {:<20} {:<20}'.format(i + 1, layer_name_function(layer), output_shape, input_shape)
                if isinstance(layer, Pool2DLayer) | isinstance(layer, Deconv2DLayer):
                    print('')

        print '\nNumber of Convolutional layers : {}'.format(
            len(filter(lambda x: isinstance(x, Conv2DLayer) | isinstance(x, Deconv2DLayer), layer_list)))
        print 'Number of parameters : {}'.format(np.sum(map(np.size, get_all_param_values(self.output_layer))))
        print('-' * 75)


if __name__ == '__main__':
    Network(input_shape=(5, 3, 224, 224)).summary()
