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
   num_frames = input:size()[1]
   frame_size = input:size()[2]
   num_output_frames = 1 + math.floor((num_frames - kW)/dW)
   self.output = torch.Tensor(num_output_frames, frame_size)
   frame_number = 0
   while frame_number < num_output_frames do
   	 frame_number = frame_number + 1
   	 kernel_top = math.min(input:size()[2], kernel_bottom + self.kW - 1)
	 window = input[{{kernel_bottom, kernel_top}, {}}]:clone()
	 res = torch.sum(window:mul(self.beta), 1):exp()
	 res:mul(1/(kernel_top - kernel_bottom + 1)):log():mul(1/self.beta)
	 self.output[frame_number] = res	 
   end
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
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
