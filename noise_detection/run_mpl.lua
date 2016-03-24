-- author = ray
require 'torch'
require 'optim'

-- configuration
enable_cuda = true
enable_plot = true
num_threads = 4
manual_seed = 1
desired_acc = 0.99
max_epoch = 100
-- configuration for prep_data
data_path = 'data/training/training_set.dat'
train_size = 14000
train_mean_err = 1e-5
train_std_err = 1e-5
test_mean_err = 0.2
test_std_err = 1.0
feat_dim = 400
train_set = nil
test_set = nil
data_mean = nil
data_std = nil
-- configuration for prep_model
num_layer = 4
num_neuron = { feat_dim,feat_dim,feat_dim,3 }
enable_dropout = false
dropout_input = 0.1
dropout_hidden = 0.2
model = nil
criterion = nil
-- configuration for train_model
log_path = 'log/'
train_log = 'mlp_3c_train.log'
test_log = 'mlp_3c_test.log'
model_file = 'model/mlp_3c.net'
enable_noise = false
noise_scale = 1e-5
learning_rate = 1e-3
weight_decay = 0
momentum = 0
learning_rate_decay = 1e-7
classes = {'env','noise','speech'}
batch_size = 10
optim_method = optim.sgd
optim_state = nil
train_logger = nil
test_logger = nil
confusion = nil
parameters = nil
gradients = nil
epoch = nil
averaged_params = nil
train_logger = nil
averaged_params = nil
test_logger = nil
-- configuration for test_model
test_acc = 0
cached_params = nil


----------------------------------------------------------------------

print('=> start running mlp: '..os.date('%Y-%m-%d %H:%M:%S'))

if enable_cuda then require 'cunn' end

-- recommended by torch for numerical operations
torch.setdefaulttensortype('torch.DoubleTensor')
-- running with multi-threads
torch.setnumthreads(num_threads)
-- use fixed seed for repeatable experiments
torch.manualSeed(manual_seed)

-- pre-processing data
dofile('prep_data.lua')
-- constructing model
dofile('prep_model.lua')
-- defining training procedure
dofile('train_model.lua')
-- defining test procedure
dofile('test_model.lua')

repeat
	train()
	test()
until test_acc>desired_acc or epoch>max_epoch

print('=> finish running mlp: '..os.date('%Y-%m-%d %H:%M:%S'))
