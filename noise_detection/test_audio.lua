-- author = ray

require 'torch'
require 'nn'
require 'audio'
require 'cunn'
require 'optim'

-- configuration
test_path = 'data/test'
model_file = 'model/mlp_3c.net'
target_sr = 16000
enable_cuda = true

-- load the mlp model if running locally
loaded = torch.load(model_file)
print('=> testing audios with mlp model')
model = loaded.model
data_mean = loaded.mean
data_std = loaded.std
print(model)

--[[
-- loop all the audios inside the test folder
for file in io.popen('find '..test_path..' -type f -name \'*.wav\''):lines() do
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
	-- save prediction result to an Audacity label file
	local label_file = io.open(string.gsub(file,'wav','label'),'w')

	-- fetch 25ms audio every 10ms as test sampels
	-- run MLP on the sample to get predicted label
	local ed = math.floor((signal:numel()-400)/160)-1
	local pred_possibilities = torch.Tensor(ed,2)
	local pred_labels = torch.Tensor(ed)
	for i=1,ed do
		-- get test sample from original audio
		local test_sample = signal[{ {i*160-159,i*160+240} }]
		-- normalize the test sample
		test_sample = test_sample:csub(loaded.mean):cdiv(loaded.std)
		print(test_sample:mean())
		print(test_sample:std())
		os.exit()
		-- transform to cuda if required
		test_sample = enable_cuda and test_sample:cuda() or test_sample:double()
		-- predict result with mlp model
		local pred = model:forward(test_sample)
		-- apply exponential function to the result to get the possibilities
		pred = pred:exp():float()
		pred_possibilities[i] = pred
		local pos,lab = torch.max(pred,1)
		pred_labels[i] = lab[1]-1
		-- save label if sample is predicted noise
		if pred_labels[i]==1 then
			label_file:write(string.format('%f\t%f\tnoise\n',(i-1)*0.01,(i-1)*0.01+0.025))
		end
		--label_file:write(string.format('%f\t%f\t%d\n',pred_possibilities[i][1],pred_possibilities[i][2],pred_labels[i]))
	end
	label_file:close()
end
--]]

-- loop all the audios inside the test folder
for file in io.popen('find '..test_path..' -type f -name \'*.wav\''):lines() do
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
	local pred_possibilities = torch.Tensor(ed,2)
	local pred_labels = torch.Tensor(ed)
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
	print('mean: '..test_samples:mean())
	print('std: '..test_samples:std())
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
		end
		-- if label==2 then
		-- 	label_file:write(string.format('%f\t%f\tspeech\n',(i-1)*0.01,(i-1)*0.01+0.025))
		-- end
		--label_file:write(string.format('%f\t%f\t%f\t%f\n',pred[1],pred[2],pred[3],label))
	end
	label_file:close()
	pos_file:close()
end



--[[
tmp_file = 'data/tmp_data.dat'
train_set = torch.load(tmp_file)
print(train_set)
train_set.data = train_set.data:cuda()

-- classes = classes or {'neg','pos'}
-- confusion = optim.ConfusionMatrix(classes)
-- test_set = train_set
-- test_set.data:cuda()
-- dofile('test_model.lua')
-- test()

for i=1,train_set.size do
	pred = model:forward(train_set.data[i])
	pred:exp():float()
	local pos,lab = torch.max(pred,1)
	label = lab[1]-1
	if train_set.labels[i]~=label then
		print('mismatch: sample #'..i)
		os.exit()
	end
	-- print(pred)
	-- print(label)
	-- print(train_set.labels[i])
end

print('good')
--]]
