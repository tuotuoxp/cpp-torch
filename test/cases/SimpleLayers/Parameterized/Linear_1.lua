require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

input = torch.Tensor(6,1024)

net = nn.Linear(1024,256)
net:evaluate()

output = net:forward(input)

--print(output:size())

torch.save(arg[1], input)
torch.save(arg[2], output)
  
net:clearState()
torch.save(arg[3], net)