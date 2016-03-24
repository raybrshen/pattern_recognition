-- author = ray
require 'torch'
require 'audio'
require 'lfs'

print('================================================================================')
print('<= '..arg[0]..os.date(' %Y-%m-%d %H:%M:%S'))

-- configurations
local target_sr = 16000  -- sampling rate
local num_peak = 600
local dataset_size = num_peak*3
local positive_path = 'data/training/noise/'
local positive_wave = 'data/training/concatenated/noise_concatenated.wav'
--local negative_wave = 'data/training/speech_concatenated.wav'
local label_path = 'data/training/concatenated/noise_concatenated.label'
local data_path = 'data/training/dataset_noise.dat'
local label_tag = 'noise'


print('=> concatenating audios')
local positive_signal = nil
for file in io.popen('find '..positive_path..' -type f -name \'*.wav\''):lines() do
	local signal, sr = audio.load(file)
	-- print file name and number of frames
	print('| '..file..' ('..signal:numel()..')')
	-- check frame rate
	if sr~=target_sr then
		print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
		os.exit()
	end
	-- save wave signals
	if positive_signal==nil then
		positive_signal = signal
	else
		positive_signal = torch.cat(positive_signal,signal,1)
	end
end

print('=> saving concatenated audio to: '..positive_wave)
audio.save(positive_wave, positive_signal, target_sr)

-- convert the data matrix to a vector
positive_signal = positive_signal[{ {},1 }]
-- number of data points
local signal_len = positive_signal:numel()
-- mark if the point has been used
local flag_arr = torch.Tensor(signal_len):fill(0)
-- log start and end time of a training example
local selected = torch.Tensor(dataset_size,2)
-- get three training example from each peak
local positive_feats = torch.Tensor(dataset_size,400)

-- sort wave array along 1st dimension descendingly
sorted_signal,idx_arr = torch.sort(positive_signal,1,true)

print('=> processing concatenated noise signal')
-- loop from highest peak to get noise indexes
local idx = 0
local peak_idx = 0
for i=1,num_peak do
	-- get the peak index that is not yet used
	repeat
		idx = idx+1
		peak_idx = idx_arr[idx]
	until flag_arr[peak_idx]==0
	-- get the left and right indexes of 400 points
	local left = peak_idx-199
	local right = peak_idx+200
	-- save the left and right indexes' time as Audacity labels
	selected[i*3-2][1] = (left-40)/16000  -- time in seconds
	selected[i*3-2][2] = (right-40)/16000
	selected[i*3-1][1] = left/16000
	selected[i*3-1][2] = right/16000
	selected[i*3][1] = (left+40)/16000
	selected[i*3][2] = (right+40)/16000
	-- save the features to feature vectors
	positive_feats[i*3-2] = positive_signal[{ {left,right} }]
	positive_feats[i*3-1] = positive_signal[{ {left-40,right-40} }]
	positive_feats[i*3] = positive_signal[{ {left-40,right-40} }]
	-- mark the the range of [peak-15ms,peak+80ms] as used
	flag_arr[{ {peak_idx-240,peak_idx+1280} }] = 1
end

print('=> feature extraction finished')
print('| dataset size: '..dataset_size)

print('=> saving Audacity labels to: '..label_path)
local label_file = io.open(label_path,'w')
for i=1,dataset_size do
	label_file:write(string.format('%f\t%f\t%s\n',selected[i][1],selected[i][2],label_tag))
end
label_file:close()

print('=> saving data to: '..data_path)
local data = {
	feats = positive_feats,
	labels = torch.ones(dataset_size),
	size = dataset_size
}
torch.save(data_path,data)

--[[
-- get and save features of negative examples from wave file
print('=> processing audio of negative examples')
local signal, sr = audio.load(negative_wave)
-- print file name and number of frames
print('| '..negative_wave..' ('..signal:numel()..')')
-- check frame rate
if sr~=target_sr then
	print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
	os.exit()
end
signal = signal[{ {},1 }]
local ed = math.floor((signal:numel()-400)/160)-1
local negative_feats = torch.Tensor(ed,400)
for i=1,ed do
	negative_feats[i] = signal[{ {i*160-159,i*160+240} }]
end

print('=> saving binary training data to: '..training_set)
-- print the data to be saved
print('| positive samples: '..(#positive_feats)[1])
print('| negative samples: '..(#negative_feats)[1])
-- concatenate positive and negative examples
local data = torch.cat(positive_feats,negative_feats,1)
local labels = torch.cat(torch.ones((#positive_feats)[1]),torch.zeros((#negative_feats)[1]),1)
-- shuffle the whole training set
local shuffle = torch.randperm((#labels)[1]):type('torch.LongTensor')
data = data:index(1,shuffle)
labels = labels:index(1,shuffle)
-- save to a binary file
local training = {data=data,labels=labels}
torch.save(training_set,training)
--]]

print('<= '..arg[0]..os.date(' %Y-%m%d %H:%M:%S'))
