-- author = ray

require 'torch'
require 'audio'
require 'cunn'

model_path = 'model/mlp_3c.net'
target_sr = 16000
enable_cuda = true

-- load the mlp model if running locally
local loaded = torch.load(model_path)
local model = loaded.model
local data_mean = loaded.mean
local data_std = loaded.std

from_dir = '/home/user/Desktop/Experiment/ASR_Noise_Filtering/test_20151113_clap_mix_16000/danilo/'
to_dir = '/home/user/Desktop/Experiment/ASR_Noise_Filtering/test_20151113_clap_filter_mlp/danilo/'
for file in io.popen('find '..from_dir..' -type f -name \'*.wav\''):lines() do
	-- load the audio file
	local signal, sr = audio.load(file)
	-- print file name and number of frames
	print('| '..file..' ('..signal:numel()..')')
	-- check frame rate
	if sr~=target_sr then
		print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
		os.exit()
	end
	-- transform to 1 dimension
	signal = signal[{ {},1 }]

	-- fetch 25ms audio every 10ms as test sampels
	-- run MLP on the sample to get predicted label
	local ed = math.floor((signal:numel()-400)/80)-1
	local noise_flag = torch.Tensor(signal:numel()):fill(1):byte()
	local test_samples = torch.Tensor(ed,400)
	for i=1,ed do
		-- get test sample from original audio
		test_samples[i] = signal[{ {(i-1)*80+1,(i-1)*80+400} }]
	end

	-- normalize the test sample
	for i=1,400 do
		test_samples[{ {},i }]:add(-data_mean[i])
		test_samples[{ {},i }]:div(data_std[i])
	end

	-- transform to cuda if required
	test_samples = enable_cuda and test_samples:cuda() or test_samples:double()
	-- save to label file
	local label_file = io.open(string.gsub(file,'wav','txt'),'w')
	local pos_file = io.open(string.gsub(file,'wav','pos'),'w')
	-- run prediction for each sample
	for i=1,ed do
		pred = model:forward(test_samples[i])
		pred:exp():float()
		local pos,lab = torch.max(pred,1)
		label = lab[1]-1
		-- print(pred)
		-- print(label)
		-- os.exit()
		if label==1 then
			label_file:write(string.format('%f\t%f\tnoi-%.2f\n',(i-1)*0.01,(i-1)*0.01+0.025,pred[2]))
			pos_file:write(string.format('%f\t%f\t%f\t%f\n',pred[1],pred[2],pred[3],label))
			noise_flag[{ {(i-1)*80+1,(i-1)*80+400} }] = 0
		end
	end

	local filtered_signal = signal:maskedSelect(noise_flag)
	filtered_signal = filtered_signal:view(filtered_signal:numel(),1)
	local to_file = to_dir..paths.basename(file)
	audio.save(to_file,filtered_signal,target_sr)

	label_file:close()
	pos_file:close()
end



from_dir = '/home/user/Desktop/Experiment/ASR_Noise_Filtering/test_20151113_clap_mix_16000/randy/'
to_dir = '/home/user/Desktop/Experiment/ASR_Noise_Filtering/test_20151113_clap_filter_mlp/randy/'
for file in io.popen('find '..from_dir..' -type f -name \'*.wav\''):lines() do
	-- load the audio file
	local signal, sr = audio.load(file)
	-- print file name and number of frames
	print('| '..file..' ('..signal:numel()..')')
	-- check frame rate
	if sr~=target_sr then
		print('=> frame rate error: expected '..target_sr..' but get '..sr..')')
		os.exit()
	end
	-- transform to 1 dimension
	signal = signal[{ {},1 }]

	-- fetch 25ms audio every 10ms as test sampels
	-- run MLP on the sample to get predicted label
	local ed = math.floor((signal:numel()-400)/80)-1
	local noise_flag = torch.Tensor(signal:numel()):fill(1):byte()
	local test_samples = torch.Tensor(ed,400)
	for i=1,ed do
		-- get test sample from original audio
		test_samples[i] = signal[{ {(i-1)*80+1,(i-1)*80+400} }]
	end

	-- normalize the test sample
	for i=1,400 do
		test_samples[{ {},i }]:add(-data_mean[i])
		test_samples[{ {},i }]:div(data_std[i])
	end

	-- transform to cuda if required
	test_samples = enable_cuda and test_samples:cuda() or test_samples:double()
	-- save to label file
	local label_file = io.open(string.gsub(file,'wav','txt'),'w')
	local pos_file = io.open(string.gsub(file,'wav','pos'),'w')
	-- run prediction for each sample
	for i=1,ed do
		pred = model:forward(test_samples[i])
		pred:exp():float()
		local pos,lab = torch.max(pred,1)
		label = lab[1]-1
		-- print(pred)
		-- print(label)
		-- os.exit()
		if label==1 then
			label_file:write(string.format('%f\t%f\tnoi-%.2f\n',(i-1)*0.01,(i-1)*0.01+0.025,pred[2]))
			pos_file:write(string.format('%f\t%f\t%f\t%f\n',pred[1],pred[2],pred[3],label))
			noise_flag[{ {(i-1)*80+1,(i-1)*80+400} }] = 0
		end
	end

	local filtered_signal = signal:maskedSelect(noise_flag)
	filtered_signal = filtered_signal:view(filtered_signal:numel(),1)
	local to_file = to_dir..paths.basename(file)
	audio.save(to_file,filtered_signal,target_sr)

	label_file:close()
	pos_file:close()
end
