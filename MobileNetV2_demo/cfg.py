import os


# model config
bn_eps = 1e-3
bn_momentum = 0.99

first_conv_name = 'conv1'

# data preprocess
image_mean = [127.5, 127.5, 127.5]
image_std = [127.5, 127.5, 127.5]

model_path = 'train_log/model_dump'
log_path = 'train_log/logs'

if not os.path.exists('train_log'):
    os.mkdir('train_log')
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

dtype = 'float32'


# data preprocess
fb_preprocess = False
if fb_preprocess:
    init_model_checkpoint_path = 'train_log/model_fb.ckpt'
else:
    init_model_checkpoint_path = 'train_log/model.ckpt'

# data augmentation
inception_resize = False
mix_up = False
multi_size = False
min_size = 256
max_size = 512
image_size = 224

batch_size = 256
num_gpus = 4
data_num_threads = 20
eval_batch_size = 200
train_num = 1281127

val_num = 50000
max_epoch = 30
lr_decay_step = 10

timeline_log = False

init_lr = 0.0001
weight_decay = 1e-4
log_step = 20
val_step = 100

batch_norm_decay = 0.997
batch_norm_center = True
batch_norm_scale = True

sysmetric = True
layer_wise = False
only_shift = False

