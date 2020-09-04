base_architecture = 'vgg19_bn'
img_size = 448
prototype_shape = (1000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '448_1000prototypes_distloss'

data_path = '/usr/xtmp/cfchen/datasets/cub200/'
train_dir = data_path + 'train_augmented/'
test_dir = data_path + 'test/'
train_eval_dir = data_path + 'train/'
train_push_dir = '/usr/xtmp/cfchen/datasets/cub200/train/'
train_batch_size = 60
test_batch_size = 60
train_push_batch_size = 60

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'dist': 0.5
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
