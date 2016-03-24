-- author = ray
require 'torch'
require 'xlua'
require 'optim'

-- configuration
test_acc = test_acc or 0
cached_params = nil

----------------------------------------------------------------------

print '=> defining test procedure'
function test()
  -- mark epoch start time
  local time = sys.clock()
  -- check if averaged params are used
  if averaged_params then
    cached_params = parameters:clone()
    parameters:copy(averaged_params)
  end
  -- set model to evaluate mode
  -- for modules that differ in training and testing, like Dropout
  model:evaluate()

  -- test over test data
  print('=> testing on test set')
  for t = 1,test_set.size do
    -- display progress
    xlua.progress(t, test_set.size)
    -- get new samples
    local input = test_set.data[t]
    input = enable_cuda and input:cuda() or input:double()
    local target = test_set.labels[t]+1
    -- test sample
    local pred = model:forward(input)
    confusion:add(pred, target)
  end

	-- calculate time elapsed
  time = sys.clock()-time
  time = time/test_set.size
  print("| time to test one sample = "..(time*1000)..'ms')
  -- print confusion matrix
  print(confusion)
  test_acc = confusion.totalValid

  -- update log/plot
  test_logger:add{['mean class accuracy (test set)'] = confusion.totalValid*100}
  if enable_plot then
    test_logger:style{['mean class accuracy (test set)'] = '-'}
    test_logger:plot()
  end

  -- check if averaged params are used
  if averaged_params then
    -- restore parameters
    parameters:copy(cached_params)
  end

  -- prepare for next iteration
  confusion:zero()
end

