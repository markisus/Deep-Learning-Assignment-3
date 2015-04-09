require 'nn'
require 'A3_skeleton'

m = nn.TemporalLogExpPooling(1,1,1)
t = torch.Tensor(1,1):fill(0)
o = m:forward(t)