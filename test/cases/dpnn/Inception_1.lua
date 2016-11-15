require 'torch'
require 'nn'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')

input = torch.Tensor(6, 192, 96, 96)
config = {inputSize = 192,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 32},
     reduceSize = {96, 16, 32, 64},
     pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
     batchNorm = true}
net = nn.Inception(config)
net:evaluate()

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)