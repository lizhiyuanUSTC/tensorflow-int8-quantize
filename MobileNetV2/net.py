import tensorflow as tf
from tensorflow.python.training import moving_averages
import math
import pickle
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

import cfg
from layer import conv_bn_relu, add
from quantize_model import prepare_calibrate_imgs, find_weight_scale, find_feature_map_scale


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def create_separable_conv(x, out_channels, ksize, stride=1,
                          qweight=False, qactivation=False, scale=None,
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
                         qactivation=qactivation, padding='SAME', scale=scale,
                         has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                         scope=prefix + 'expand')
        depthwise_filters = in_channels * 6
    f = conv_bn_relu(f, depthwise_filters, ksize, stride=stride, qweight=qweight, qactivation=qactivation,
                     padding='SAME', groups=depthwise_filters, scale=scale,
                     has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                     scope=prefix + 'depthwise')
    f = conv_bn_relu(f, pointwise_filters, 1, stride=1, qweight=qweight, qactivation=qactivation,
                     padding='SAME', scale=scale, has_bn=has_bn, has_relu=False, phase_train=phase_train,
                     scope=prefix + 'project')

    if in_channels == pointwise_filters and stride == 1:
        f = add(f, x, phase_train, qactivation=qactivation, scope=prefix)
    return f


def inference(images, phase_train=False, has_bn=True, image_norm=True, alpha=1.0,
              qactivation=False, qweight=False, scale=None):
    images = tf.cast(images, dtype=cfg.dtype)
    if image_norm:
        mean = np.reshape(np.array(cfg.image_mean), (1, 1, 1, 3))
        std = np.reshape(np.array(cfg.image_std), (1, 1, 1, 3))
        images = (images - mean) / std
    else:
        images = images - 128

    first_block_filters = _make_divisible(32 * alpha, 8)

    f = conv_bn_relu(images, first_block_filters, 3, 2, qweight=qweight, qactivation=qactivation, scale=scale,
                     has_bn=has_bn, has_relu=True, phase_train=phase_train, scope=cfg.first_conv_name)
    f = create_separable_conv(f, int(16 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=0)

    f = create_separable_conv(f, int(24 * alpha), 3, 2, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=1)
    f = create_separable_conv(f, int(24 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=2)

    f = create_separable_conv(f, int(32 * alpha), 3, 2, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=3)
    f = create_separable_conv(f, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=4)
    f = create_separable_conv(f, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=5)

    f = create_separable_conv(f, int(64 * alpha), 3, 2, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=6)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=7)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=8)
    f = create_separable_conv(f, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=9)

    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=10)
    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=11)
    f = create_separable_conv(f, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=12)

    f = create_separable_conv(f, int(160 * alpha), 3, 2, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=13)
    f = create_separable_conv(f, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=14)
    f = create_separable_conv(f, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=15)

    f = create_separable_conv(f, int(320 * alpha), 3, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    f = conv_bn_relu(f, last_block_filters, 1, 1, qweight=qweight, qactivation=qactivation, scale=scale,
                     has_bn=has_bn, has_relu=True, phase_train=phase_train, scope='conv_1')

    f = conv_bn_relu(f, 1000, 1, stride=1, padding='SAME', qweight=qweight, qactivation=False, scale=scale,
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

        saver = tf.train.Saver(tf.get_collection('params'))
        scale_dict = OrderedDict()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            if os.path.exists('scale'):
                with open('scale', 'rb') as f:
                    scale_dict = pickle.load(f)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                name = cfg_nodes[i]['name']
                node = nodes[i]
                scale_dict[name] = {}

                if cfg_nodes[i][name]['type'] == 'Conv2D':
                    weight = node['W']
                    cfg_nodes[i]['W'] = weight
                    print(name, 'weights', weight.max(), weight.min())
                    scale = find_weight_scale(weight)
                    print(name, 'weights', scale.shape)
                    scale_dict[name]['W'] = scale

                    biases = node['b']
                    cfg_nodes[i]['b'] = biases

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

            print(cfg_nodes)
            with open('cfg_nodes.pkl', 'wb') as f:
                pickle.dump(cfg_nodes, f)


def merge_bn_params():
    param_list = []
    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=True, has_bn=True)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')

        model_checkpoint_path = 'train_log/model.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                name = cfg_nodes[i]['name']
                node = nodes[i]
                if cfg_nodes[i]['type'] == 'Conv2D':
                    print(node, node['W'])
                    node[i]['W'], node[i]['b'] = node['W'], node['b']
                    if cfg.first_conv_name == name:
                        node[i]['W'], node[i]['b'] = fix_input(node[i]['W'], node[i]['b'])
                param_list.append(cfg_nodes[i]['W'])
                param_list.append(cfg_nodes[i]['b'])
    return param_list


def init_merge_bn():
    param_list = merge_bn_params()
    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=False, has_bn=False)

        model_checkpoint_path = 'train_log/model_merge_bn.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=val_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm(len(var_list)):
                print(var_list[i].shape, param_list[i].shape)
                sess.run(tf.assign(var_list[i], param_list[i]))

            saver.save(sess, model_checkpoint_path, write_meta_graph=False,
                       write_state=False)


def find_connect():
    val_graph = tf.Graph()
    with val_graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=False, has_bn=False, qweight=False, qactivation=False)
    nodes = tf.get_collection('nodes')
    cfg_nodes = tf.get_collection('cfg_nodes')

    for i in range(len(cfg_nodes)):

        if cfg_nodes[i]['type'] == 'Conv2D':
            print(nodes[i]['input'].name, nodes[i]['output'].name)
            output = nodes[i]['output']
            for j in range(len(cfg_nodes)):
                input = nodes[j]['input']
                if cfg_nodes[j]['type'] == 'Conv2D':
                    if output.name == input.name:
                        cfg_nodes[j]['input_layer'] = cfg_nodes[i]['name']


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
    # init_merge_bn()
    # evaluate(model_checkpoint_path='train_log/model_merge_bn.ckpt', has_bn=False,
    #          qweight=True, qactivation=True, image_norm=False)
    find_connect()


if __name__ == '__main__':
    main()
