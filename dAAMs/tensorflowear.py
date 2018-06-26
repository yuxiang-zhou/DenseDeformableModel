import tensorflow as tf
import numpy as np
import menpo.io as mio
import sys
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path
import traceback


batch_size = 32

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0002,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
# tf.app.flags.DEFINE_integer('batch_size', batch_size, '''The batch size to use.''')
# tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
#                             '''How many preprocess threads to use.''')
# tf.app.flags.DEFINE_string('train_dir', 'ckpt/ear_train',
#                            '''Directory where to write event logs '''
#                            '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/vol/atlas/homes/gt108/Projects/ibugface/pretrained_models/resnet_v1_50.ckpt',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
# tf.app.flags.DEFINE_integer('max_steps', 100000,
#                             '''Number of batches to run.''')
# tf.app.flags.DEFINE_string('train_device', '/gpu:1',
#                            '''Device to train with.''')
# tf.app.flags.DEFINE_string('dataset_path', '', 'Dataset directory')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(2, channels)

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)



def preprocess_for_train(image, is_training=True):
  """Preprocesses the given image for training.
  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)
  if is_training:
      image = distort_color(image)
  image *= 255
  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

class Dataset(object):
    def __init__(self, name, root, batch_size=1):
        self.name = name
        self.root = Path(root)
        self.batch_size = batch_size

    def get_keys(self, path='images'):
        path = self.root / path
        keys = [x.stem for x in path.glob('*')]
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

class EarDB(Dataset):
    def __init__(self, batch_size=64, db_name='WPUTEDB-train', shape=(250, 190), num_classes=500, root='/homes/yz4009/wd/PickleModel/EarRecognition/',is_training=True):
        self.name = db_name
        self.batch_size = batch_size
        self.root = Path(root)
        self.dataset = mio.import_pickle(str(self.root / '{}.pkl'.format(db_name)), encoding='latin1')
        self.num_classes=num_classes
        self.shape = shape
        self.is_training = is_training

    def get_keys(self, path='images'):
        path = self.root / path
        keys = list(map(str, np.arange(len(self.dataset))))
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def get_images(self, key, shape=None):
        def wrapper(index):
            pixels = self.dataset[int(index)][1].resize(self.shape).pixels_with_channels_at_back()
            if len(pixels.shape) == 2:
                pixels = np.dstack([pixels, pixels, pixels])
            tmp_pixels = (pixels).astype(np.float32)

            return tmp_pixels

        image = tf.py_func(wrapper, [key],
                                   [tf.float32])[0]

        image.set_shape(self.shape + (3,))
        return image

    def get_labels(self, key, shape=None):
        def wrapper(index):
            return np.int32(self.dataset[int(index)][0])

        label = tf.py_func(wrapper, [key],
                                   [tf.int32])[0]

        label = tf.one_hot(label, self.num_classes, dtype=tf.int32)
        label.set_shape([self.num_classes,])
        return label, None

    def get(self, *names):
        producer = tf.train.string_input_producer(self.get_keys(),
                                                  shuffle=True)
        key = producer.dequeue()
        images = self.get_images(key)
        images = preprocess_for_train(images,is_training=self.is_training)

        image_shape = tf.shape(images)
        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape)
            tensors.append(label)

        return tf.train.shuffle_batch(tensors,
                              self.batch_size,
                              capacity=2000, min_after_dequeue=200)


def network(inputs, scale=1, output_classes=500, is_training=True):
    with slim.arg_scope(nets.resnet_utils.resnet_arg_scope(is_training=is_training, weight_decay=0.05)):
        net, layers = nets.resnet_v1.resnet_v1_50(inputs, output_classes)
    return net[:, 0, 0, :], layers

def train(log_dir='tf',batch_size=64, db_name='WPUTEDB-train', shape=(250, 190), num_classes=500, root='/homes/yz4009/wd/PickleModel/EarRecognition/'):
    logging.set_verbosity(0)
    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        provider = EarDB(batch_size=batch_size,db_name=db_name, shape=shape, num_classes=num_classes,root=root)
        images, labels = provider.get('labels')

        tf.image_summary('images', images)
        # Define model graph.
        prediction, _ = network(images, output_classes=num_classes)

        # Add a smoothed l1 loss to every scale and the combined output.
        crossentropy_loss = slim.losses.softmax_cross_entropy(prediction, labels)

        global_step = slim.get_or_create_global_step()

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('total loss', total_loss)

        tf.scalar_summary('crossentropy loss', crossentropy_loss)

        regularisation_loss = tf.reduce_mean(
            slim.losses.get_regularization_losses())

        tf.scalar_summary('regularisation loss', regularisation_loss)

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step,
            2000, FLAGS.learning_rate_decay_factor, staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

    with tf.Session(graph=g) as sess:

        saver = tf.train.Saver()
        if FLAGS.pretrained_model_checkpoint_path:
            saver = tf.train.Saver([v for v in tf.trainable_variables() if 'logits' not in v.name and not 'adam' in v.name.lower()])
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
              MOVING_AVERAGE_DECAY, global_step)

        update_ops.append(variable_averages.apply(moving_average_variables))

        train_op = slim.learning.create_train_op(
            total_loss, optimizer, update_ops=update_ops)

        logging.set_verbosity(1)

        print("Start Training...")


        slim.learning.train(train_op,
                            'ckpt/' + log_dir  + '_train/',
                            # number_of_steps=1700,
                            save_summaries_secs=60,
                            save_interval_secs=600)


def test(log_dir='tf',batch_size=64, db_name='WPUTEDB-test', shape=(250, 190), num_classes=500, root='/homes/yz4009/wd/PickleModel/EarRecognition/'):
    test_provider = EarDB(batch_size=batch_size,db_name=db_name, shape=shape, num_classes=num_classes,root=root,is_training=False)
    images, labels = test_provider.get('labels')

    tf.image_summary('images', images)

    predictions, _ = network(images, is_training=False, output_classes=num_classes)

    predictions = tf.to_int32(tf.argmax(predictions, 1))
    labels = tf.to_int32(tf.argmax(labels, 1))

    tf.scalar_summary('accuracy', slim.metrics.accuracy(predictions, labels))

    num_batches = 859 // batch_size

    sess = tf.Session()

    # These are streaming metrics which compute the "running" metric,
    # e.g running accuracy
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        "streaming_accuracy": slim.metrics.streaming_accuracy(predictions, labels),
    })

    # Define the streaming summaries to write:
    for metric_name, metric_value in metrics_to_values.items():
        tf.scalar_summary(metric_name, metric_value)

    global_step = slim.get_or_create_global_step()
    variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
    variables_to_restore = variable_averages.variables_to_restore(
      slim.get_model_variables())
    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '',
        'ckpt/' + log_dir  + '_train/',
        'ckpt/' + log_dir  + '_eval/',
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        summary_op=tf.merge_all_summaries(),
        variables_to_restore=variables_to_restore,
        eval_interval_secs=30)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tenserflow Experiments')
    parser.add_argument('-t', dest='is_test', default=False, help='Train/Test Switch', action="store_true")
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='Batch_size')
    parser.add_argument('--root', dest='root_dir', default='/homes/yz4009/wd/PickleModel/EarRecognition/', help='Root DB Directory')
    parser.add_argument('--db', dest='db_name', default='WPUTEDB', help='DB Name')
    parser.add_argument('--ncls', dest='num_classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--log', dest='log_dir', default='tf', help='log directories')
    parser.add_argument('--shape', dest='shape', type=int, nargs=2, default=[225,225], help='Image Shape')
    args = parser.parse_args()
    print(args)
    # return
    #
    db_name=args.db_name
    shape=tuple(args.shape)
    num_classes=args.num_classes

    batch_size=args.batch_size
    root=args.root_dir
    log_dir = args.log_dir

    if args.is_test:
        test(log_dir=log_dir,batch_size=batch_size,db_name=db_name+'-test', shape=shape, num_classes=num_classes,root=root)
    else:
        while True:
            try:
                train(log_dir=log_dir,batch_size=batch_size,db_name=db_name+'-train', shape=shape, num_classes=num_classes,root=root)
            except Exception as e:
                print(e)
                traceback.print_exc()
                pass
