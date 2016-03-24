-- author = ray
require 'torch'

-- configuration
local concat_dataset_path = 'data/training/training_set.dat'
local data_list = {
	'data/training/dataset_env.dat',
	'data/training/dataset_clap.dat',
	'data/training/dataset_noise.dat',
	'data/training/dataset_speech.dat'
}

-- concatenating files
print('=> concatenating files')
local feats, labels, size = nil,nil,nil
for i=1,#data_list do
	local loaded = torch.load(data_list[i])
	print('| '..data_list[i]..' [size:'..loaded.size..']')
	feats = feats and torch.cat(feats,loaded.feats,1) or loaded.feats
	labels = labels and torch.cat(labels,loaded.labels,1) or loaded.labels
	size = size and size+loaded.size or loaded.size
end

print('=> shuffling the whole data set')
local shuffle = torch.randperm(size):type('torch.LongTensor')
feats = feats:index(1,shuffle)
labels = labels:index(1,shuffle)

-- print result
local dataset = {
	feats=feats,
	labels=labels,
	size=size
}
print('=> concatenated result')
print(dataset)

print('=> saving dataset to: '..concat_dataset_path)
torch.save(concat_dataset_path,dataset)
