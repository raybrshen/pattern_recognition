-- author = ray
require 'torch'
require 'audio'
require 'lfs'
require 'xlua'

print('================================================================================')
print('=> '..arg[0]..os.date(' %Y-%m-%d %H:%M:%S'))

-- environment wave file

-- configurations
local target_sr = 16000  -- sampling rate
local speech_dir_path = 'data/training/speech/'
local speech_wave_path = 'data/training/speech/long_speech_2.wav'
local speech_label_path = 'data/training/speech/long_speech_2.label'
local speech_data_path = 'data/training/dataset_speech.dat'
local env_wave_path = 'data/training/env/environment.wav'
local env_data_path = 'data/training/dataset_env.dat'
local label_tag = 'speech'


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
	local ten_ms = math.floor((signal:numel()-400)/160)-1
	-- calculate the max amplitude of each 10ms'
	local max_amp = torch.Tensor(ten_ms)
	for i=1,ten_ms do
		local tmp_tensor = signal[{ {(i-1)*160+1,(i-1)*160+400} }]
		max_amp[i] = torch.max(tmp_tensor)-torch.min(tmp_tensor)
	end
	return max_amp
end

print('=> calculating threshold for environment sound: ')
local env_signal = load_audio(env_wave_path)
local env_max_amp = get_max_amp(env_signal)
local threshold = torch.max(env_max_amp)
print(string.format('| threshold: %.2e',threshold))

--[[
print('=> concatenating audios')
local speech_signal = nil
for file in io.popen('find '..speech_dir_path..' -type f -name \'*.wav\''):lines() do
	local signal, sr = audio.load(file)
	-- print file name and number of frames
	print('| '..file..' ('..signal:numel()..')')
	-- check frame rate
	if sr~=target_sr then
		print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
		os.exit()
	end
	-- save wave signals
	if speech_signal==nil then
		speech_signal = signal
	else
		speech_signal = torch.cat(speech_signal,signal,1)
	end
end
print('=> saving concatenated audio to: '..speech_wave_path)
audio.save(speech_wave_path, speech_signal, target_sr)
-- convert the data matrix to a vector
speech_signal = speech_signal[{ {},1 }]
--]]
---[[
-- for concatenated audio
print('=> loading concatenated audio: '..speech_wave_path)
local speech_signal = load_audio(speech_wave_path)
--]]

local speech_max_amp = get_max_amp(speech_signal)
local max_amp_size = speech_max_amp:numel()

print('=> separating speech and env samples')
local speech_feats = nil
local env_feats = nil
-- save to Audacity label file
local label_file = io.open(speech_label_path,'w')
-- extract speech and env samples from speech signal base on the threshold
for i=1,max_amp_size do
	local sample = speech_signal[{ {(i-1)*160+1,(i-1)*160+400} }]:view(1,400)
	if speech_max_amp[i]>threshold then
		speech_feats = speech_feats and torch.cat(speech_feats,sample,1) or sample
		label_file:write(string.format('%f\t%f\t%s\n',(i-1)*0.01,(i-1)*0.01+0.025,label_tag))
	else
		env_feats = env_feats and torch.cat(env_feats,sample,1) or sample
	end
end
label_file:close()
num_env_samples = (#env_feats)[1]
num_speech_samples = (#speech_feats)[1]
print('| env samples: '..num_env_samples)
print('| speech samples: '..num_speech_samples)


print('=> saving env dataset to: '..env_data_path)
local env_data = {
	feats = env_feats,
	labels = torch.zeros(num_env_samples),
	size = num_env_samples
}
torch.save(env_data_path,env_data)
print('=> saving speech dataset to: '..speech_data_path)
local speech_data = {
	feats = speech_feats,
	labels = torch.Tensor(num_speech_samples):fill(2),
	size = num_speech_samples
}
torch.save(speech_data_path,speech_data)


print('<= '..arg[0]..os.date(' %Y-%m%d %H:%M:%S'))
