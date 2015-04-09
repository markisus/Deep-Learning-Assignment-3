require 'nn'
require 'A3_skeleton'

m = nn.TemporalLogExpPooling(5,3,7)
t = torch.linspace(1,100):resize(100,1)
o = m:forward(t)