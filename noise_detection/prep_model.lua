-- author = ray
require 'torch'
require 'nn'

-- configuration
feat_dim = feat_dim or 400
num_layer = num_layer or 4  -- including input/output layer
num_neuron = num_neuron or { feat_dim,feat_dim,feat_dim,2 }
enable_dropout = enable_dropout or false
dropout_input = dropout_input or 0.1
dropout_hidden = dropout_hidden or 0.2
model = nil
criterion = nil

----------------------------------------------------------------------

print('=> constructing mlp model')
model = nn.Sequential()
if enable_dropout then model:add(nn.Dropout(dropout_input)) end
--model:add(nn.Reshape(feat_dim))
-- adding hidden layers
for i=2,num_layer-1 do
	model:add(nn.Linear(num_neuron[i-1],num_neuron[i]))
	--model:add(nn.BatchNormalization(num_neuron[i-1]))
	model:add(nn.ReLU())
	if enable_dropout then model:add(nn.Dropout(dropout_hidden)) end
end
-- adding output layer
model:add(nn.Linear(num_neuron[num_layer-1],num_neuron[num_layer]))

print('=> defining nll loss function')
-- using negative log likelihood criterion
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

-- print the model
print(model)
-- print the loss function
print(criterion)

