-- author = ray
require 'torch'

-- configuration
data_path = data_path or 'data/training/training_set.dat'
feat_dim = feat_dim or 400
train_size = train_size or 8000
train_mean_err = train_mean_err or 1e-5
train_std_err = train_std_err or 1e-5
test_mean_err = test_mean_err or 0.2
test_std_err = test_std_err or 1.0
train_set = nil
train_set = nil
data_mean = nil
data_std = nil

----------------------------------------------------------------------

print('=> loading data: '..data_path)
local loaded = torch.load(data_path)
train_set = {
	data = loaded.feats[{ {1,train_size},{} }],
	labels = loaded.labels[{ {1,train_size} }],
	size = train_size }
test_set = {
	data = loaded.feats[{ {train_size+1,-1},{} }],
	labels = loaded.labels[{ {train_size+1,-1} }],
	size = loaded.size-train_size }
if feat_dim~=(#train_set.data)[2] then
	print('=> feature dimension unmatch')
	os.exit()
end
print('| feature dimension: '..feat_dim)
print('| training set size: '..train_set.size)
print('| test set size: '..test_set.size)

print('=> nomalizing data')
data_mean = torch.Tensor(feat_dim)
data_std = torch.Tensor(feat_dim)
-- calculate the means/stds of each feature using training data
-- normalize training and test data using the same means/stds
for i=1,feat_dim do
	data_mean[i] = train_set.data[{ {},i }]:mean()
	data_std[i] = train_set.data[{ {},i }]:std()
	train_set.data[{ {},i }]:add(-data_mean[i])
	train_set.data[{ {},i }]:div(data_std[i])
	test_set.data[{ {},i }]:add(-data_mean[i])
	test_set.data[{ {},i }]:div(data_std[i])
end

print('=> verifying normalization result')
local verify=torch.Tensor(4,feat_dim)
for i=1,feat_dim do
	-- verify mean of training data is closed to 0
	verify[1][i] = math.abs(train_set.data[{ {},i }]:mean())
	-- verify standard deviation of training data is closed to 1
	verify[2][i] = math.abs(train_set.data[{ {},i }]:std()-1)
	-- verify mean of test data is closed to 0
	verify[3][i] = math.abs(test_set.data[{ {},i }]:mean())
	-- verify standard deviation of test data is closed to 1
	verify[4][i] = math.abs(test_set.data[{ {},i }]:std()-1)
end
local tmp_func = function(b) if b then print('pass') else print('fail') os.exit() end end
io.write('| training data mean: ') tmp_func(verify[1]:max()<train_mean_err)
io.write('| training data std: ') tmp_func(verify[2]:max()<train_std_err)
io.write('| test data mean: ') tmp_func(verify[3]:max()<test_mean_err)
io.write('| test data std: ') tmp_func(verify[4]:max()<test_std_err)

