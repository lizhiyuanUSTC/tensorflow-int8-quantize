import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
import pickle
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

import cfg
from quantize_model import prepare_calibrate_imgs, find_weight_scale, find_feature_map_scale

scale = None
if os.path.exists('scale'):
    with open('scale', 'rb') as file:
        scale = pickle.load(file)


def int_quantize(x, scale_factor, num_bits=8, phase_train=False):
    max_int = 2 ** (num_bits - 1) - 1

    if phase_train:
        x_scale = x / scale_factor
        x_int = tf.stop_gradient(tf.round(x_scale) - x_scale) + x_scale
    else:
        x_int = tf.round(x / scale_factor)
    x_clip = tf.clip_by_value(x_int, -max_int - 1, max_int)
    return x_clip * scale_factor


def compute_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
        return fan_in, fan_out
    elif len(shape) == 4:
        ksize = shape[0] * shape[1]
        fan_in = shape[2] * ksize
        fan_out = shape[3] * ksize
        return fan_in, fan_out
    else:
        raise NotImplementedError


def _variable_on_cpu(name, shape, initializer, trainable=True, dtype=cfg.dtype):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the Variable
      shape: list of ints
      initializer: initializer of Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, dtype=cfg.dtype):
    fan_in, fan_out = compute_fans(shape)
    stddev = math.sqrt(1.0 / fan_in)
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=cfg.dtype),
                           dtype=dtype)
    if cfg.weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.weight_decay, name='weight_loss')
        if cfg.dtype == 'float16':
            weight_decay = tf.to_float(weight_decay)
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_norm_for_conv(x, phase_train, scope='bn'):
    channels = x.shape.as_list()[3]
    with tf.variable_scope(scope):
        gamma = _variable_on_cpu('gamma', [channels, ], tf.constant_initializer(1.0), dtype='float32')
        beta = _variable_on_cpu('beta', [channels, ], tf.constant_initializer(0.0), dtype='float32')
        moving_mean = _variable_on_cpu('moving_mean', [channels, ], dtype='float32',
                                       initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = _variable_on_cpu('moving_variance', [channels, ], dtype='float32',
                                           initializer=tf.zeros_initializer(), trainable=False)
        tf.add_to_collection('params', gamma)
        tf.add_to_collection('params', beta)
        tf.add_to_collection('params', moving_mean)
        tf.add_to_collection('params', moving_variance)

        if not phase_train:
            normed_x, _, _ = tf.nn.fused_batch_norm(x, gamma, beta,
                                                    mean=moving_mean, variance=moving_variance,
                                                    is_training=False, epsilon=cfg.bn_eps)
        else:
            normed_x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                                                                     is_training=True, epsilon=cfg.bn_eps)

            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, cfg.bn_momentum)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_var, cfg.bn_momentum)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        return normed_x, [x, moving_mean, moving_variance, beta, gamma]


def conv_bn_relu(x, out_channels, ksize, stride=1, groups=1, qweight=False, qactivation=False,
                 padding='SAME',
                 has_bn=True, has_relu=True, phase_train=False,
                 scope=None):
    node = {'input': x}

    cfg_node = {'name': scope,
                'type': 'Conv2D',
                'out': out_channels,
                'ksize': ksize,
                'stride': stride,
                'groups': groups,
                'padding': padding,
                'active': has_relu}

    with tf.variable_scope(scope):
        in_channels = x.shape.as_list()[3]
        cfg_node['in'] = in_channels

        assert in_channels % groups == 0 and out_channels % groups == 0
        shape = [ksize, ksize, in_channels // groups, out_channels]
        kernel = _variable_with_weight_decay('W', shape)
        tf.add_to_collection('params', kernel)
        node['W'] = kernel
        if qweight:
            kernel = int_quantize(kernel, scale[scope]['W'], num_bits=8, phase_train=phase_train)

        if groups == 1:
            f = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
        else:
            if out_channels == groups and in_channels == groups:
                f = tf.nn.depthwise_conv2d(x,
                                           tf.transpose(kernel, (0, 1, 3, 2)),
                                           [1, stride, stride, 1],
                                           padding=padding)
            else:
                kernel_list = tf.split(kernel, groups, axis=3)
                x_list = tf.split(x, groups, axis=3)
                f = tf.concat(
                    [tf.nn.conv2d(x_list[i], kernel_list[i], [1, stride, stride, 1], padding=padding)
                     for i in range(groups)], axis=3)

        if has_bn:
            f, bn_info = batch_norm_for_conv(f, phase_train)
            _, moving_mean, moving_variance, beta, gamma = bn_info
            s = gamma / tf.sqrt(moving_variance + cfg.bn_eps)
            node['W'] = kernel * tf.reshape(s, (1, 1, 1, -1))
            node['b'] = beta - s * moving_mean
        else:
            biases = _variable_on_cpu('b', out_channels, tf.constant_initializer(0.0))
            tf.add_to_collection('params', biases)
            node['b'] = biases

            f = tf.nn.bias_add(f, biases)

        if has_relu:
            f = tf.nn.relu6(f)
        node['output'] = f
        print(scope, f.shape)

        tf.add_to_collection('nodes', {scope: node})
        tf.add_to_collection('cfg_nodes', {scope: cfg_node})

        if qactivation:
            f = int_quantize(f, scale[scope]['output'], num_bits=8, phase_train=phase_train)
        return f


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def create_separable_conv(x, out_channels, ksize, stride=1, qweight=False, qactivation=False,
                          padding='SAME',
                          has_bn=True, has_relu=True, phase_train=False,
                          block_id=0, ):
    in_channels = x.shape.as_list()[3]
    depthwise_filters = in_channels
    pointwise_filters = _make_divisible(out_channels, 8)
    prefix = 'block_{}_'.format(block_id)

    f = x
    if block_id:
        # Expand
        f = conv_bn_relu(x, in_channels * 6, 1, stride=1, qweight=qweight,
                         qactivation=qactivation, padding=padding,
                         has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                         scope=prefix + 'expand')
        depthwise_filters = in_channels * 6
    f = conv_bn_relu(f, depthwise_filters, ksize, stride=stride, qweight=qweight, qactivation=qactivation,
                     padding=padding, groups=depthwise_filters,
                     has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                     scope=prefix + 'depthwise')
    f = conv_bn_relu(f, pointwise_filters, 1, stride=1, qweight=qweight, qactivation=qactivation, padding=padding,
                     has_bn=has_bn, has_relu=False, phase_train=phase_train,
                     scope=prefix + 'project')

    if in_channels == pointwise_filters and stride == 1:
        f = add(f, x, phase_train, qactivation=qactivation, scope=prefix)
    return f


def add(x, y, phase_train, has_relu=False, qactivation=False, scope=None):
    f = x + y
    if has_relu:
        f = tf.nn.relu6(f)
    node = {'input': [x, y],
            'output': f}
    cfg_node = {'name': scope,
                'type': 'Add'}
    tf.add_to_collection('nodes', {scope: node})
    tf.add_to_collection('cfg_nodes', {scope: cfg_node})
    if qactivation:
        f = int_quantize(f, scale[scope]['output'], num_bits=8, phase_train=phase_train)

    return f


def inference(images, phase_train=False, has_bn=True, image_norm=True, alpha=1.0,
              qactivation=False, qweight=False):
    images = tf.cast(images, dtype=cfg.dtype)
    if image_norm:
        mean = np.reshape(np.array(cfg.image_mean), (1, 1, 1, 3))
        std = np.reshape(np.array(cfg.image_std), (1, 1, 1, 3))
        images = (images - mean) / std
    else:
        images = images - 128

    first_block_filters = _make_divisible(32 * alpha, 8)

    f = conv_bn_relu(images, first_block_filters, 3, 2, qweight=qweight, qactivation=qactivation,
                     has_bn=has_bn, has_relu=True, phase_train=phase_train, scope=cfg.first_conv_name)
    f = create_separable_conv(f, int(16 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=0)

    f = create_separable_conv(f, int(24 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=1)
    f = create_separable_conv(f, int(24 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=2)

    f = create_separable_conv(f, int(32 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=3)
    f = create_separable_conv(f, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=4)
    f = create_separable_conv(f, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=5)

    f = create_separable_conv(f, int(64 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=6)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=7)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=8)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=9)

    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=10)
    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=11)
    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=12)

    f = create_separable_conv(f, int(160 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=13)
    f = create_separable_conv(f, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=14)
    f = create_separable_conv(f, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=15)

    f = create_separable_conv(f, int(320 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    f = conv_bn_relu(f, last_block_filters, 1, 1, qweight=qweight, qactivation=qactivation,
                     has_bn=has_bn, has_relu=True, phase_train=phase_train, scope='conv_1')

    f = conv_bn_relu(f, 1000, 1, stride=1, padding='SAME', qweight=qweight, qactivation=False,
                     has_bn=False, has_relu=False, phase_train=phase_train, scope='prediction')
    f = tf.reduce_mean(f, axis=[1, 2], keepdims=False)
    if cfg.dtype == 'float16':
        f = tf.cast(f, dtype='float32')

    return f


def init():
    from keras.applications import MobileNetV2

    network = MobileNetV2(alpha=1.0)
    params = network.get_weights()

    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)

        inference(images, False)

        model_checkpoint_path = 'train_log/model.ckpt'
        var_list = tf.get_collection('params')
        assert len(var_list) == len(params)
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(len(var_list)):
                if 'depthwise' in var_list[i].name and len(params[i].shape) == 4:
                    params[i] = np.transpose(params[i], (0, 1, 3, 2))
                if len(params[i].shape) == 2:
                    params[i] = np.expand_dims(params[i], 0)
                    params[i] = np.expand_dims(params[i], 0)
                print(var_list[i].name, var_list[i].shape, params[i].shape)
                sess.run(tf.assign(var_list[i], params[i]))

            saver.save(sess, model_checkpoint_path, write_meta_graph=False,
                       write_state=False)


def fix_input(w, b):
    mean = np.array(cfg.image_mean, dtype=np.float32)
    std = np.array(cfg.image_std, dtype=np.float32)
    w = w / np.reshape(std, (1, 1, -1, 1))
    _, k_h, k_w, _ = w.shape

    graph = tf.Graph()
    with graph.as_default():
        mean = tf.constant(mean)
        mean = tf.reshape(mean, (1, 1, 1, 3))
        mean = tf.tile(mean, (1, k_h, k_w, 1))
        conv_mean = tf.nn.conv2d(mean - 128, w, strides=[1, 1, 1, 1], padding='VALID')
        b = b - tf.squeeze(conv_mean)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            b = sess.run(b)
    return w, b


def find_quantize_scale(model_checkpoint_path):
    graph = tf.Graph()
    with graph.as_default():
        images = prepare_calibrate_imgs()

        _ = inference(images, False, has_bn=False, image_norm=False)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')
        cfg_dict = OrderedDict()
        for item in tqdm(cfg_nodes):
            name = list(item.keys())[0]
            node = item[name]
            cfg_dict[name] = node

        saver = tf.train.Saver(tf.get_collection('params'))
        scale_dict = {}

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            if os.path.exists('scale'):
                with open('scale', 'rb') as f:
                    scale_dict = pickle.load(f)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                item = nodes[i]
                name = list(item.keys())[0]
                node = item[name]

                scale_dict[name] = {}

                if cfg_nodes[i][name]['type'] == 'Conv2D':
                    weight = node['W']
                    cfg_dict[name]['W'] = weight
                    print(name, 'weights', weight.max(), weight.min())
                    scale = find_weight_scale(weight)
                    print(name, 'weights', scale.shape)
                    scale_dict[name]['W'] = scale

                    biases = node['b']
                    cfg_dict[name]['b'] = biases

                inputs = node['input']
                if isinstance(inputs, list):
                    scale_dict[name]['input'] = []
                    for _inputs in inputs:
                        print(name, 'inputs', _inputs.max(), _inputs.min())
                        scale = find_feature_map_scale(_inputs)
                        scale_dict[name]['input'].append(scale)
                else:
                    print(name, 'inputs', inputs.max(), inputs.min())
                    if name == cfg.first_conv_name:
                        scale = 1.0
                    else:
                        scale = find_feature_map_scale(inputs)
                    scale_dict[name]['input'] = scale

                outputs = node['output']
                print(name, 'outputs', outputs.max(), outputs.min())
                scale = find_feature_map_scale(outputs)
                scale_dict[name]['output'] = scale

            with open('scale', 'wb') as f:
                pickle.dump(scale_dict, f)

            print(cfg_dict)
            with open('cfg.pkl', 'wb') as f:
                pickle.dump(cfg_dict, f)


def merge_bn():
    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=True, has_bn=True)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')
        cfg_dict = OrderedDict()
        for item in tqdm(cfg_nodes):
            name = list(item.keys())[0]
            node = item[name]
            cfg_dict[name] = node

        model_checkpoint_path = 'train_log/model.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                item = nodes[i]
                name = list(item.keys())[0]
                node = item[name]
                if cfg_dict[name]['type'] == 'Conv2D':
                    print(node, node['W'])
                    cfg_dict[name]['W'], cfg_dict[name]['b'] = node['W'], node['b']

    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)

        inference(images, False, image_norm=False, has_bn=False)
        nodes = tf.get_collection('nodes')

        model_checkpoint_path = 'train_log/model_merge_bn.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm(range(len(nodes))):
                item = nodes[i]
                name = list(item.keys())[0]
                node = item[name]
                if cfg_dict[name]['type'] == 'Conv2D':
                    print(node, node['W'])
                    if name == cfg.first_conv_name:
                        cfg_dict[name]['W'], cfg_dict[name]['b'] = fix_input(cfg_dict[name]['W'], cfg_dict[name]['b'])
                    sess.run(tf.assign(node['W'], cfg_dict[name]['W']))
                    sess.run(tf.assign(node['b'], cfg_dict[name]['b']))

            saver.save(sess, model_checkpoint_path, write_meta_graph=False,
                       write_state=False)


def evaluate(model_checkpoint_path='train_log/model.ckpt', has_bn=True,
             qweight=False, qactivation=False, image_norm=True):
    import dataset
    val_graph = tf.Graph()
    with val_graph.as_default():
        iterator = dataset.make_val_dataset()
        images, labels = iterator.get_next()
        val_logits = inference(images, False, has_bn=has_bn, image_norm=image_norm,
                               qweight=qweight, qactivation=qactivation)
        val_acc = 100 * tf.reduce_mean(tf.cast(tf.nn.in_top_k(val_logits, labels, 1), dtype=tf.float32))

        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            eval_acc = 0
            num_epoch = 50000 // cfg.eval_batch_size
            num_epoch = 10
            for _ in tqdm(range(num_epoch)):
                _val_acc = sess.run(val_acc)
                eval_acc += _val_acc
            print(eval_acc / num_epoch)
            return eval_acc / num_epoch


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # init()
    # evaluate(model_checkpoint_path='train_log/model.ckpt', has_bn=True,
    #          qweight=False, qactivation=False, image_norm=True)
    # merge_bn()
    evaluate(model_checkpoint_path='train_log/model_merge_bn.ckpt', has_bn=False,
             qweight=True, qactivation=True, image_norm=False)


if __name__ == '__main__':
    main()
