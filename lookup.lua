require 'torch'

function lookup(word, vector)
    vector:fill(0)
    handle = io.popen("python /scratch/ml4133/query_word.py \""..word.."\"")
    vector_string = handle:read("*a")
    i = 1
    for entry in vector_string:gmatch("%S+") do
    	vector[i] = tonumber(entry)
	i = i+1
    end
    return vector
end