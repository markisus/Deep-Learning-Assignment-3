require 'torch'
require 'nn'
require 'lookup'


model = torch.load('/scratch/ml4133/a3_model.net', 'ascii')

cmd = torch.CmdLine()

cmd:option('-input', ' ')

params = cmd:parse(arg)
review = params.input

opt = {}
opt.words_per_review = 100
opt.inputDim = 50

            vectorized_document = torch.Tensor(opt.words_per_review, opt.inputDim):fill(0)
	    vector = torch.Tensor(opt.inputDim):fill(0)
	    word_number = 1
	    for word in review:gmatch("%S+") do
  	        word = word:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
	    	if word_number <= opt.words_per_review then
		    vectorized_document[word_number] = lookup(word, vector)
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

vectorized_document = vectorized_document:resize(1, 100, 50)
pred = model:forward(vectorized_document)
_, argmax = pred:max(2)
print(argmax[1][1])