require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

inputSize = 3
outputSize = 2

input = torch.Tensor(inputSize, 7, 7)

net = nn.DepthConcat(1);
net:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
net:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
net:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)