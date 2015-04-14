require 'torch'
require 'nn'
require 'optim'
require 'xlua'

ffi = require('ffi')

glove_cache = {}

function preprocess_data(raw_data, opt)
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.words_per_review, opt.inputDim)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    vector = torch.Tensor(opt.inputDim)

    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
	    --xlua.progress((i-1)*(opt.nTrainDocs+opt.nTestDocs) + j, opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            vectorized_document = torch.Tensor(opt.words_per_review, opt.inputDim):fill(0)
	    word_number = 1
	    for word in document:gmatch("%S+") do
  	        word = word:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
	    	if word_number <= opt.words_per_review then
		    if glove_cache[word] == nil then
		       goto continue
		    end
		    vectorized_document[word_number] = glove_cache[word]
		else
		    break
		end
		word_number = word_number + 1
		::continue::
            end
	    words_used = word_number - 1
	    if words_used < opt.words_per_review and words_used ~= 0 then
	       for t = 1, opt.words_per_review - words_used do
	       	   --print(t+words_used, ((t-1) % words_used), t)
		   fake_word_index = t + words_used
		   substitute_word_index = ((t-1) % words_used) + 1
		   if pcall(function () vectorized_document[fake_word_index] = vectorized_document[substitute_word_index] end) then
		   --ok
		   else
			print(fake_word_index, substitute_word_index, t, words_used)
		   end
   	       	   
	       end
	    end
	    
            data[k] = vectorized_document
            labels[k] = i
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            print("epoch: ", epoch, " batch: ", batch)
        end

        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)

    end
end

function test_model(model, data, labels, opt)
    
    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end

function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

function main()

    -- Configuration parameters
    opt = {}
    opt.words_per_review = 100
    -- change these to the appropriate data locations
    opt.glovePath = "/scratch/ml4133/glove.6B.50d.txt" -- path to raw glove data .txt file
    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- word vector dimensionality
    opt.inputDim = 50 --From now on don't change this because the database is built with 50 dimensional vectors
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 9600
    opt.nTestDocs = 400
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.01
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)

    print("Loading glove...")
    glove_cache = load_glove(opt.glovePath, opt.inputDim)

    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, opt)
    print(processed_data:size())
    -- split data into makeshift training and validation sets
    local training_data = processed_data[{{1, opt.nTrainDocs * opt.nClasses}, {}, {}}]:clone()
    local training_labels = labels[{{1, opt.nTrainDocs * opt.nClasses}}]:clone()
    
    -- make your own choices - here I have not created a separate test set
    local test_data = processed_data[{{opt.nTrainDocs * opt.nClasses + 1, (opt.nTrainDocs + opt.nTestDocs)*opt.nClasses}, {}, {}}]:clone()
    local test_labels = labels[{{opt.nTrainDocs * opt.nClasses + 1, (opt.nTrainDocs + opt.nTestDocs)*opt.nClasses}}]:clone()

    -- construct model:
    model = nn.Sequential()
    
    model:add(nn.Reshape(opt.words_per_review*opt.inputDim, true))
    model:add(nn.Linear(opt.words_per_review*opt.inputDim, 5))
    model:add(nn.LogSoftMax())
    criterion = nn.ClassNLLCriterion()
   
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    local results = test_model(model, test_data, test_labels)
    print(results)
end

main()
