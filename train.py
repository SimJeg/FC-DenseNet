#!/usr/bin/env python
import argparse
import os
import shutil
import sys
import time
from datetime import datetime
import imp
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.regularization import regularize_network_params
from lasagne.layers import get_output
from theano.tensor.shared_randomstreams import RandomStreams
from data_loader import load_data

from metrics import theano_metrics, crossentropy


def batch_loop(iterator, f, epoch, phase, history):
    """ Loop on the batches """

    n_batches = iterator.get_n_batches()
    n_imgs = 0.

    for i in range(n_batches):
        X, Y = iterator.next()
        batch_size = X.shape[0]
        n_imgs += batch_size

        loss, I, U, acc = f(X, Y[:, None, :, :])
        if i == 0:
            loss_tot = loss * batch_size
            I_tot = I
            U_tot = U
            acc_tot = acc * batch_size
        else:
            loss_tot += loss * batch_size
            I_tot += I
            U_tot += U
            acc_tot += acc * batch_size

        # # Progression bar ( < 74 characters)
        sys.stdout.write('\rEpoch {} : [{} : {}%]'.format(epoch, phase, int(100. * (i + 1) / n_batches)))
        sys.stdout.flush()

    history[phase]['loss'].append(loss_tot / n_imgs)
    history[phase]['jaccard'].append(np.mean(I_tot / U_tot))
    history[phase]['accuracy'].append(acc_tot / n_imgs)

    return history


def train(cf):

    ###############
    #  load data  #
    ###############

    print('-' * 75)
    print('Loading data')
    #TODO ; prepare a public version of the data loader
    train_iter, val_iter, test_iter = load_data(cf.dataset,
                                                train_crop_size=cf.train_crop_size,
                                                batch_size=cf.batch_size,
                                                horizontal_flip=True,
                                                )

    n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_labels()

    print('Number of images : train : {}, val : {}, test : {}'.format(
        train_iter.get_n_samples(), val_iter.get_n_samples(), test_iter.get_n_samples()))

    ###################
    #   Build model   #
    ###################

    # Build model and display summary    
    net = cf.net
    net.summary()

    # Restore
    if hasattr(cf, 'pretrained_model'):
        print('Using a pretrained model : {}'.format(cf.pretrained_model))
        net.restore(cf.pretrained_model)

    # Compile functions
    print('Compilation starts at ' + str(datetime.now()).split('.')[0])
    params = lasagne.layers.get_all_params(net.output_layer, trainable=True)
    lr_shared = theano.shared(np.array(cf.learning_rate, dtype='float32'))
    lr_decay = np.array(cf.lr_sched_decay, dtype='float32')

    # Create loss and metrics
    for key in ['train', 'valid']:

        # LOSS
        pred = get_output(net.output_layer, deterministic=key == 'valid',
                          batch_norm_update_averages=False, batch_norm_use_averages=False)
        loss = crossentropy(pred, net.target_var, void_labels)

        if cf.weight_decay:
            weightsl2 = regularize_network_params(net.output_layer, lasagne.regularization.l2)
            loss += cf.weight_decay * weightsl2

        # METRICS
        I, U, acc = theano_metrics(pred, net.target_var, n_classes, void_labels)

        # COMPILE
        start_time_compilation = time.time()
        if key == 'train':
            updates = cf.optimizer(loss, params, learning_rate=lr_shared)
            train_fn = theano.function([net.input_var, net.target_var], [loss, I, U, acc], updates=updates)
        else:
            val_fn = theano.function([net.input_var, net.target_var], [loss, I, U, acc])

        print('{} compilation took {:.3f} seconds'.format(key, time.time() - start_time_compilation))

    ###################
    #    Main loops   #
    ###################

    # metric's sauce
    init_history = lambda: {'loss': [], 'jaccard': [], 'accuracy': []}
    history = {'train': init_history(), 'val': init_history(), 'test': init_history()}
    patience = 0
    best_jacc_val = 0
    best_epoch = 0

    if hasattr(cf, 'pretrained_model'):
        print('Validation score before training')
        print batch_loop(val_iter, val_fn, 0, 'val', {'val': init_history()})

    # Training main loop
    print('-' * 30)
    print('Training starts at ' + str(datetime.now()).split('.')[0])
    print('-' * 30)

    for epoch in range(cf.num_epochs):

        # Train
        start_time_train = time.time()
        history = batch_loop(train_iter, train_fn, epoch, 'train', history)
        # Validation
        start_time_valid = time.time()
        history = batch_loop(val_iter, val_fn, epoch, 'val', history)

        # Print
        out_str = \
            '\r\x1b[2 Epoch {} took {}+{} sec. ' \
            'loss = {:.5f} | jacc = {:.5f} | acc = {:.5f} || ' \
            'loss = {:.5f} | jacc = {:.5f} | acc = {:.5f}'.format(
                epoch, int(start_time_valid - start_time_train), int(time.time() - start_time_valid),
                history['train']['loss'][-1], history['train']['jaccard'][-1], history['train']['accuracy'][-1],
                history['val']['loss'][-1], history['val']['jaccard'][-1], history['val']['accuracy'][-1])

        # Monitoring jaccard
        if history['val']['jaccard'][-1] > best_jacc_val:
            out_str += ' (BEST)'
            best_jacc_val = history['val']['jaccard'][-1]
            best_epoch = epoch
            patience = 0
            net.save(os.path.join(cf.savepath, 'model.npz'))
        else:
            patience += 1

        print out_str

        np.savez(os.path.join(cf.savepath, 'errors.npz'), metrics=history, best_epoch=best_epoch)

        # Learning rate scheduler
        lr_shared.set_value(lr_shared.get_value() * lr_decay)

        # Finish training if patience has expired or max nber of epochs reached
        if patience == cf.max_patience or epoch == cf.num_epochs - 1:
            # Load best model weights
            net.restore(os.path.join(cf.savepath, 'model.npz'))

            # Test
            print('Training ends\nTest')
            if test_iter.get_n_samples() == 0:
                print 'No test set'
            else:
                history = batch_loop(test_iter, val_fn, epoch, 'test', history)

                print ('Average cost test = {:.5f} | jacc test = {:.5f} | acc_test = {:.5f} '.format(
                    history['test']['loss'][-1],
                    history['test']['jaccard'][-1],
                    history['test']['accuracy'][-1]))

                np.savez(os.path.join(cf.savepath, 'errors.npz'), metrics=history, best_epoch=best_epoch)

            # Exit
            return


def initiate_training(cf):

    # Seed : to make experiments reproductible, use deterministic convolution in CuDNN with THEANO_FLAGS
    np.random.seed(cf.seed)
    theano.tensor.shared_randomstreams.RandomStreams(cf.seed)

    if not os.path.exists(cf.savepath):
        os.makedirs(cf.savepath)
    else:
        stop = raw_input('\033[93m The following folder already exists {}. '
                         'Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf.savepath))
        if stop == 'n':
            return

    print('-' * 75)
    print('Config\n')
    print('Local saving directory : ' + cf.savepath)
    print('Model path : ' + cf.model_path)

    # We also copy the model and the training scipt to reproduce exactly the experiments
    shutil.copy('train.py', os.path.join(cf.savepath, 'train.py'))
    shutil.copy(os.path.join('models', cf.model_path), os.path.join(cf.savepath, 'model.py'))
    shutil.copy(cf.config_path, os.path.join(cf.savepath, 'cf.py'))

    # Train
    train(cf)


if __name__ == '__main__':

    # To launch an experiment, use the following command line :
    # THEANO_FLAGS='device=cuda,optimizer=fast_compile,optimizer_including=fusion' python train.py -c config_path -e experiment_name
    # Logs of the training will be stored in the folder cf.savepath/experiment_name

    parser = argparse.ArgumentParser(description='DenseNet training')

    parser.add_argument('-c', '--config_path',
                        type=str,
                        default=None,
                        help='Configuration file')

    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of the experiment')
    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration path using ' \
                                              '-c config/pathname in the command line'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment using -e name in the command line'

    # Parse the configuration file
    cf = imp.load_source('config', arguments.config_path)
    cf.savepath = arguments.exp_name
    cf.config_path = arguments.config_path

    # You can easily launch different experiments by slightly changing cf and initiate training
    initiate_training(cf)
