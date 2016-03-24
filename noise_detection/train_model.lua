-- author = ray
require 'torch'
require 'xlua'
require 'optim'

-- configuration
log_path = log_path or 'log/'
train_log = train_log or 'mlp_train.log'
test_log = test_log or 'mlp_test.log'
model_file = model_file or 'model/mlp.net'
enable_noise = enable_noise or false
noise_scale = noise_scale or 1e-5
learning_rate = learning_rate or 1e-3
weight_decay = weight_decay or 0
momentum = momentum or 0
learning_rate_decay = learning_rate_decay or 1e-7
classes = classes or {'neg','pos'}
batch_size = batch_size or 10
optim_method = optim_method or optim.sgd
optim_state = nil
train_logger = nil
test_logger = nil
confusion = nil
parameters = nil
gradients = nil
averaged_params = nil
epoch = nil

----------------------------------------------------------------------

-- enable cuda for gpu computing
if enable_cuda then
	print('=> gpu computing enabled')
	model:cuda()
	criterion:cuda()
end

-- records confussion aross classes
confusion = optim.ConfusionMatrix(classes)
-- log results to files
train_logger = optim.Logger(paths.concat(log_path, train_log))
test_logger = optim.Logger(paths.concat(log_path, test_log))

-- extract and flatterns all the trainable parameters of the model to 1-dim vector
if model then
	parameters,gradients = model:getParameters()
end

print('=> configuring optimizer')
optim_state = {
	learning_rate = learning_rate,
	weight_decay = weight_decay,
	momentum = momentum,
	learning_rate_decay = learning_rate_decay
}

print('=> defining training procedure')	
function train()
	-- epoch tracker
  epoch = epoch or 1
  -- mark epoch start time
  local time = sys.clock()
  -- set model to training mode
  -- for modules that differ in training and testing, like Dropout
  model:training()
  -- shuffle data at each epoch
  local shuffle = torch.randperm(train_set.size)

  -- do one epoch (loop the whole training set)
  print("\n => training epoch #"..epoch..' [batchSize='..batch_size..']')
  for t = 1,train_set.size,batch_size do
		-- display progress
		xlua.progress(t, train_set.size)
	  -- create mini batch
	  local inputs = {}
	  local targets = {}
	  for i = t,math.min(t+batch_size-1,train_set.size) do
	    -- load new samples
	    local input = train_set.data[shuffle[i]]
	    local target = train_set.labels[shuffle[i]]+1
	    -- add random noise to prevent over-fitting (should not be too much)
	    if enable_noise then input:add(torch.rand(400):csub(0.5):mul(noise_scale)) end
	    -- transform to cuda type if required
	    input = enable_cuda and input:cuda() or input:double()
	    table.insert(inputs, input)
	    table.insert(targets, target)
	  end

	  -- create closure to evaluate the cost f(X) and gradients (df/dW)
	  local feval = function(x)
	  	-- get new parameters
	    if x~=parameters then
	       parameters:copy(x)
	    end
	    -- reset gradients
	    gradients:zero()
	    -- cost is the average of all criterions
	    local cost = 0
	    -- evaluate function for complete mini batch
	    for i = 1,#inputs do
	      -- estimate the cost
	      local output = model:forward(inputs[i])
	      local err = criterion:forward(output, targets[i])
	      cost = cost + err
	      -- estimate partial derivatives
	      local df_dW = criterion:backward(output, targets[i])
	      model:backward(inputs[i], df_dW)
	      -- update confusion
	      confusion:add(output, targets[i])
	    end
	    -- normalize the cost and gradients
	    cost = cost/(#inputs)
	    gradients:div(#inputs)
	    -- return the cost and gradients
	    return cost,gradients
	  end

	  -- optimize on current mini-batch
	  if optim_method == optim.asgd then
	    _,_,averaged_params = optim_method(feval, parameters, optim_state)
	  else
	    optim_method(feval, parameters, optim_state)
	  end
	end

	-- calculate time elapsed
	time = sys.clock()-time
	time = time/train_set.size
	print("| time to learn one sample = "..(time*1000)..'ms')
	-- print confusion matrix
	print(confusion)

	-- update logger/plot
	train_logger:add{['mean class accuracy (train set)'] = confusion.totalValid*100}
	if enable_plot then
	  train_logger:style{['mean class accuracy (train set)'] = '-'}
	  train_logger:plot()
	end

	-- save current net with mean and std of training data
	os.execute('mkdir -p '..sys.dirname(model_file))
	print('| saving model to: '..model_file)
	obj = {
		mean = data_mean,
		std = data_std,
		model = model
	}
	torch.save(model_file, obj)

	-- prepare for next epoch
	confusion:zero()
	epoch = epoch + 1
end

