import theano.tensor as T
import numpy as np

def theano_metrics(y_pred, y_true, n_classes, void_labels):
    """
    Returns the intersection I and union U (to compute the jaccard I/U) and the accuracy.

    :param y_pred: tensor of predictions. shape  (b*0*1, c) with c = n_classes
    :param y_true: groundtruth, shape  (b,0,1) or (b,c,0,1) with c=1
    :param n_classes: int
    :param void_labels: list of indexes of void labels
    :return: return tensors I and U of size (n_classes), and scalar acc
    """

    # Put y_pred and y_true under the same shape
    y_true = T.flatten(y_true)
    y_pred = T.argmax(y_pred, axis=1)

    # We use not_void in case the prediction falls in the void class of the groundtruth
    for i in range(len(void_labels)):
        if i == 0:
            not_void = T.neq(y_true, void_labels[i])
        else:
            not_void = not_void * T.neq(y_true, void_labels[i])

    I = T.zeros(n_classes)
    U = T.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = T.eq(y_true, i)
        y_pred_i = T.eq(y_pred, i)
        I = T.set_subtensor(I[i], T.sum(y_true_i * y_pred_i))
        U = T.set_subtensor(U[i], T.sum(T.or_(y_true_i, y_pred_i) * not_void))

    accuracy = T.sum(I) / T.sum(not_void)

    return I, U, accuracy


def numpy_metrics(y_pred, y_true, n_classes, void_labels):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.flatten()

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~ np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy


def crossentropy(y_pred, y_true, void_labels):
    # Flatten y_true
    y_true = T.flatten(y_true)
    
    # Clip predictions

    # Create mask
    mask = T.ones_like(y_true)
    for el in void_labels:
        mask = T.switch(T.eq(y_true, el), np.int32(0), mask)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true_tmp)

    # Compute masked mean loss
    loss *= mask
    loss = T.sum(loss) / T.sum(mask).astype('float32')

    return loss
