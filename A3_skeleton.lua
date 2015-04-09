
-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------
local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   self.num_frames = input:size()[1]
   self.frame_size = input:size()[2]
   self.num_output_frames = 1 + math.floor((self.num_frames - self.kW)/self.dW)
   num_frames = self.num_frames
   frame_size = self.frame_size
   num_output_frames = self.num_output_frames

   --print("Inside update output")

   self.output = torch.Tensor(num_output_frames, frame_size)
   self.usage = torch.Tensor(num_output_frames, num_frames):zero()
   frame_number = 0
   kernel_bottom = 1
   while frame_number < num_output_frames do
   	 frame_number = frame_number + 1
   	 kernel_top = math.min(num_frames, kernel_bottom + self.kW - 1)
	 --print(kernel_bottom, kernel_top)
	 window = input[{{kernel_bottom, kernel_top}, {}}]:clone()
	 --print(window)
	 res = torch.sum(window:mul(self.beta):exp(), 1)
	 --print(res)
	 res:mul(1/(kernel_top - kernel_bottom + 1)):log():mul(1/self.beta)
	 --print(res)
	 self.output[frame_number] = res
	 self.usage[{{frame_number}, {kernel_bottom, kernel_top}}] = 1
	 kernel_bottom = kernel_bottom + self.dW
   end
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -- gradient of output wrt to input 
   -- output is num_frames by 1 + math.floor((num_frames - kW)/dW)
   num_frames = self.num_frames
   frame_size = self.frame_size
   num_output_frames = self.num_output_frames
   exp_beta_x = input:clone():mul(self.beta):exp()
   denoms = torch.mm(self.usage, exp_beta_x)
   self.gradInput = torch.Tensor(num_frames, frame_size)
   for feature_number = 1, frame_size do
       for frame_number = 1, num_frames do
       	   numerator = exp_beta_x[{frame_number, feature_number}]
	   res = self.usage[{{}, frame_number}]:clone()
	   --print("Frame ", frame_number, " usage ", res)
	   res:mul(numerator):cdiv(denoms[{{}, feature_number}])
	   -- dot product
	   --print("res ", res)
	   --print("dOutput/dFeature ", gradOutput[{{}, feature_number}])
	   self.gradInput[{frame_number, feature_number}] = res * gradOutput[{{}, feature_number}]
       end
   end
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
