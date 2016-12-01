from lasagne.layers import (
    NonlinearityLayer, Conv2DLayer, DropoutLayer, Pool2DLayer, ConcatLayer, Deconv2DLayer,
    DimshuffleLayer, ReshapeLayer, get_output, BatchNormLayer)

from lasagne.nonlinearities import linear, softmax
from lasagne.init import HeUniform

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """

    l = NonlinearityLayer(BatchNormLayer(inputs))
    l = Conv2DLayer(l, n_filters, filter_size, pad='same', W=HeUniform(gain='relu'), nonlinearity=linear,
                    flip_filters=False)
    if dropout_p != 0.0:
        l = DropoutLayer(l, dropout_p)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = Pool2DLayer(l, 2, mode='max')

    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = ConcatLayer(block_to_upsample)
    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    # Concatenate with skip connection
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution


def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, n_classes, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols, n_classes))
    l = NonlinearityLayer(l, softmax)

    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that FC-DenseNet naturally permits this multiscale approach