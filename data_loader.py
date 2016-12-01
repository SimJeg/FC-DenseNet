from dataset_loaders.images.camvid import CamvidDataset
from numpy.random import RandomState

def load_data(dataset, train_crop_size=(224, 224), one_hot=False,
              batch_size=10,
              horizontal_flip=False,
              rng=RandomState(0)):

    if isinstance(batch_size, int):
        batch_size = [batch_size] * 3

    train_iter = CamvidDataset(which_set='train',
                               batch_size=batch_size[0],
                               seq_per_video=0,
                               seq_length=0,
                               crop_size=train_crop_size,
                               horizontal_flip=horizontal_flip,
                               get_one_hot=False,
                               get_01c=False,
                               overlap=0,
                               use_threads=True,
                               rng=rng)

    val_iter = CamvidDataset(which_set='val',
                             batch_size=batch_size[1],
                             seq_per_video=0,
                             seq_length=0,
                             crop_size=None,
                             get_one_hot=False,
                             get_01c=False,
                             shuffle_at_each_epoch=False,
                             overlap=0,
                             use_threads=True,
                             save_to_dir=False)

    test_iter = CamvidDataset(which_set='test',
                              batch_size=batch_size[2],
                              seq_per_video=0,
                              seq_length=0,
                              crop_size=None,
                              get_one_hot=False,
                              get_01c=False,
                              shuffle_at_each_epoch=False,
                              overlap=0,
                              use_threads=True,
                              save_to_dir=False)

    return train_iter, val_iter, test_iter
