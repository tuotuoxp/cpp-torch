require 'torch'
require 'nn'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')

input = torch.Tensor(6,64,96,96)
net = nn.SpatialBatchNormalization(64, nil, nil, false)
net = nn.Decorator(net)
net:evaluate()

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)