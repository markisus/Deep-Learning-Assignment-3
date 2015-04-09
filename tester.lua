require 'nn'
require 'A3_skeleton'

m = nn.TemporalLogExpPooling(5,3,4)
t = torch.linspace(1,100):resize(100,1)
g = torch.ones(32,1)
o = m:forward(t)
b = m:backward(t, g)

tt = torch.randn(100, 200):mul(5)
gg = torch.ones(32, 200)
oo = m:forward(tt)
bb = m:backward(tt, gg)