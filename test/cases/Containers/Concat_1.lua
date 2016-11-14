require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

input = torch.Tensor(6,5)

net = nn.Concat(2);
net:add(nn.Linear(5,3))
net:add(nn.Linear(5,7))

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)