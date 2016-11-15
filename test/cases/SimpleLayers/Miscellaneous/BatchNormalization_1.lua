require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

input = torch.Tensor(10, 5)

net = nn.BatchNormalization(5)
net:evaluate()

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)