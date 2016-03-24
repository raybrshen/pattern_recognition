-- author = ray
require 'torch'
require 'audio'
require 'lfs'
require 'xlua'

print('================================================================================')
print('=> '..arg[0]..os.date(' %Y-%m-%d %H:%M:%S'))

-- minimum length of a cough is 100ms=10*10ms
min_len = 10
-- environment wave file
env_wave = 'data/training/environment.wav'
-- feature dimension
feat_dim = 1600

-- configurations
local num_peak = 20
local target_sr = 16000  -- sampling rate
local positive_path = 'data/training/cough/'
local positive_wave = 'data/training/cough_concatenated.wav'
local negative_wave = 'data/training/speech_concatenated.wav'
local positive_label = 'data/training/cough_concatenated.label'
local training_set = 'data/training/training_set_3.dat'

local function sub_sampling(in_signal,in_dim,out_dim)
	local kernel_size = 2
	local step_size = in_dim/out_dim
	local out_signal = torch.Tensor(out_dim)
	for i=1,out_dim do
		local left = math.floor((i-1)*step_size)+1
		local right = math.min(left+kernel_size-1,in_dim)
		out_signal[i] = in_signal[{ {left,right} }]:mean()
	end
	return out_signal
end

local function load_audio(file)
	local signal, sr = audio.load(file)
	-- print file name and number of frames
	print('| '..file..' ('..signal:numel()..')')
	-- check frame rate
	if sr~=target_sr then
		print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
		os.exit()
	end
	signal = signal[{ {},1 }]
	return signal
end

local function get_max_amp(signal)
	-- number of 10ms'
	local ten_ms = math.floor(signal:numel()/160)
	-- calculate the max amplitude of each 10ms'
	local max_amp = torch.Tensor(ten_ms)
	for i=1,ten_ms do
		local tmp_tensor = signal[{ {(i-1)*160+1,(i-1)*160+160} }]
		max_amp[i] = torch.max(tmp_tensor)-torch.min(tmp_tensor)
	end
	return max_amp
end

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

print('=> calculating threshold for environment sound: ')
local env_signal = load_audio(env_wave)
local env_max_amp = get_max_amp(env_signal)
local threshold = torch.max(env_max_amp)
print(string.format('| threshold: %.2e',threshold))

-- convert the data matrix to a vector
positive_signal = positive_signal[{ {},1 }]
local positive_max_amp = get_max_amp(positive_signal)
local max_amp_size = positive_max_amp:numel()

local positive_feats = nil
-- save to udacity label file
local label_file = io.open(positive_label,'w')
-- extract spoken noise samples based on threshold and minimum lenth
local idx = 0
for i=1,max_amp_size do
	if positive_max_amp[i]>threshold and idx==0 then idx = i end
	if positive_max_amp[i]<=threshold and idx~=0 then
		if i-idx >= min_len then
			local left,right = idx-1,i-1
			for j=0,3 do
				local t_left,t_right=left-j,right+j
				if t_left>0 and t_right<=max_amp_size then
					label_file:write(string.format('%f\t%f\tcough\n',t_left*0.01,t_right*0.01))
					local sample = positive_signal[{ {t_left*160+1,t_right*160} }]
					-- sub-sampling
					if sample:numel()~=feat_dim then sample=sub_sampling(sample,sample:numel(),feat_dim) end
					-- add to positive feats
					sample = sample:view(1,feat_dim)
					positive_feats = positive_feats and torch.cat(positive_feats,sample,1) or sample
				end
			end
		end
		idx = 0
	end
end
label_file:close()
print(#positive_feats)

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
-- transform to 1-dim vector
signal = signal[{ {},1 }]
-- define matrix to save negative samples
local negative_feats = nil
-- window size from 100ms to 300ms step every 50ms
for w_size=1600,4800,800 do
	local ed = math.floor((signal:numel()-w_size)/160)-1
	for i=1,ed do
		xlua.progress(i, train_set.size)
		local sample = signal[{ {(i-1)*160+1,(i-1)*160+w_size} }]
		-- sub-sampling
		if sample:numel()~=feat_dim then sample=sub_sampling(sample,sample:numel(),feat_dim) end
		-- add to negative feats
		sample = sample:view(1,feat_dim)
		negative_feats = negative_feats and torch.cat(negative_feats,sample,1) or sample
	end
end
print(#negative_feats)


--[[---------------------------------------------------------------------------------

-- mark if the point has been used
local flag_arr = torch.Tensor(signal_len):fill(0)

-- log start and end time of a training example
local selected = torch.Tensor(num_peak*3,2)
-- get three training example from each peak
local positive_feats = torch.Tensor(num_peak*3,400)

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
	-- get the left and right indexes
	local left = peak_idx-199
	local right = peak_idx+200
	-- save the left and right indexes' time as Audacity labels
	-- selected[i][1] = math.floor(left/160)  -- time in 10ms
	-- selected[i][2] = math.floor(right/160)
	selected[i*3-2][1] = (left-80)/16000  -- time in seconds
	selected[i*3-2][2] = (right-80)/16000
	selected[i*3-1][1] = left/16000
	selected[i*3-1][2] = right/16000
	selected[i*3][1] = (left+80)/16000
	selected[i*3][2] = (right+80)/16000
	-- save the features to feature vectors
	positive_feats[i*3-2] = positive_signal[{ {left,right} }]
	positive_feats[i*3-1] = positive_signal[{ {left-80,right-80} }]
	positive_feats[i*3] = positive_signal[{ {left-80,right-80} }]
	-- mark the the range as used
	flag_arr[{ {left-80,right+80} }] = 1
end

print('=> saving Audacity labels to: '..positive_label)
local label_file = io.open(positive_label,'w')
for i=1,num_peak*3 do
	label_file:write(string.format('%f\t%f\tclap\n',selected[i][1],selected[i][2]))
end
label_file:close()

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
