import tensorflow as tf
import pickle
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

import cfg
from models.MobileNetV2 import inference
from quantize_model import prepare_calibrate_imgs, find_weight_scale, find_feature_map_scale


def init():
    from keras.applications import MobileNetV2

    network = MobileNetV2(alpha=1.0)
    params = network.get_weights()

    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)

        inference(images, False)

        model_checkpoint_path = 'log/model_dump/model.ckpt'
        var_list = tf.get_collection('params')
        assert len(var_list) == len(params)
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
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

            if os.path.exists('log/scale'):
                with open('log/scale', 'rb') as f:
                    scale_dict = pickle.load(f)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                name = cfg_nodes[i]['name']
                node = nodes[i]
                scale_dict[name] = {}

                if cfg_nodes[i]['type'] == 'Conv2D':
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

            with open('log/scale', 'wb') as f:
                pickle.dump(scale_dict, f)

            with open('log/cfg_nodes.pkl', 'wb') as f:
                pickle.dump(cfg_nodes, f)


def merge_bn_params():
    param_list = []
    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=True, has_bn=True)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')

        model_checkpoint_path = 'log/model_dump/model.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                name = cfg_nodes[i]['name']
                node = nodes[i]
                if cfg_nodes[i]['type'] == 'Conv2D':
                    cfg_nodes[i]['W'], cfg_nodes[i]['b'] = node['W'], node['b']
                    if cfg.first_conv_name == name:
                        cfg_nodes[i]['W'], cfg_nodes[i]['b'] = fix_input(cfg_nodes[i]['W'], cfg_nodes[i]['b'])
                    param_list.append(cfg_nodes[i]['W'])
                    param_list.append(cfg_nodes[i]['b'])
    return param_list


def init_merge_bn():
    param_list = merge_bn_params()
    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=False, has_bn=False)

        model_checkpoint_path = 'log/model_dump/model_merge_bn.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm(range(len(var_list))):
                # print(var_list[i].shape, param_list[i].shape)
                sess.run(tf.assign(var_list[i], param_list[i]))

            saver.save(sess, model_checkpoint_path, write_meta_graph=False,
                       write_state=False)


def find_connect():
    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=False, has_bn=False, qweight=False, qactivation=False)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')
        for i in range(len(cfg_nodes)):
            if cfg_nodes[i]['name'] == cfg.first_conv_name:
                cfg_nodes[i]['input_layer'] = 'image'
            if cfg_nodes[i]['type'] == 'Conv2D':
                input = nodes[i]['input']
                for j in range(len(cfg_nodes)):
                    output = nodes[j]['output']
                    if output.name == input.name:
                        cfg_nodes[i]['input_layer'] = cfg_nodes[j]['name']
                        break
            elif cfg_nodes[i]['type'] == 'Add':
                input = nodes[i]['input']
                cfg_nodes[i]['input_layer'] = []
                for _input in input:
                    for j in range(len(cfg_nodes)):
                        output = nodes[j]['output']
                        if output.name == _input.name:
                            cfg_nodes[i]['input_layer'].append(cfg_nodes[j]['name'])
                            break
            else:
                raise NotImplementedError

        for node in cfg_nodes:
            print(node['name'], node['input_layer'])


def evaluate(model_checkpoint_path='log/model_dump/model.ckpt', has_bn=True,
             qweight=False, qactivation=False, image_norm=True):
    import dataset

    scale = None
    if qweight or qactivation:
        with open('log/scale', 'rb') as f:
            scale = pickle.load(f)

    graph = tf.Graph()
    with graph.as_default():
        iterator = dataset.make_val_dataset()
        images, labels = iterator.get_next()
        val_logits = inference(images, False, has_bn=has_bn, image_norm=image_norm,
                               qweight=qweight, qactivation=qactivation, scale=scale)
        val_acc = 100 * tf.reduce_mean(tf.cast(tf.nn.in_top_k(val_logits, labels, 1), dtype=tf.float32))

        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
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
    init()
    acc_original = evaluate(model_checkpoint_path='log/model_dump/model.ckpt', has_bn=True,
                            qweight=False, qactivation=False, image_norm=True)
    init_merge_bn()
    acc_merge_bn = evaluate(model_checkpoint_path='log/model_dump/model_merge_bn.ckpt', has_bn=False,
                            qweight=False, qactivation=False, image_norm=False)
    find_quantize_scale('log/model_dump/model_merge_bn.ckpt')
    acc_int = evaluate(model_checkpoint_path='log/model_dump/model_merge_bn.ckpt', has_bn=False,
                       qweight=True, qactivation=True, image_norm=False)
    find_connect()
    print('float acc = %.3f' % acc_original)
    print('float acc merge bn = %.3f' % acc_merge_bn)
    print('int acc = %.3f' % acc_int)


if __name__ == '__main__':
    main()
