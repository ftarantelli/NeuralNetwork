import torch
from torch import nn
from matplotlib import pyplot as plt

class netNeur(nn.Module):

	def __init__(self):
		super().__init__()
		self.layer1=nn.Linear(2, 4)
		self.activation1=nn.ReLU()
		self.layer2=nn.Linear(4,2)
		self.activation2=nn.Tanh()
		self.layer3=nn.Linear(2,1)
		#self.activation3=torch.norm()

	def forward(self, x):
		x=self.layer1(x)
		x=self.activation1(x)
		x=self.layer2(x)
		x=self.activation2(x)
		x=self.layer3(x)
		#x=self.activation3(x)
		return x


def circle(x):
	r = torch.norm(x)
	#theta = nn.arctan(x[0]/x[1])
	return torch.exp(-(r-3)**2)

loss_fun=nn.MSELoss()

rete=netNeur()

losses = []
for i in range(0,5000):
	p = torch.randn((100,2))*10
	target=circle(p)
	result=rete.forward(p)

	loss=loss_fun(target, result)
	losses.append(loss.item())
	loss.backward()
	#print(loss)
	for p in rete.parameters():
		p.data.sub_(0.001*p.grad)

	rete.zero_grad()

plt.plot(losses)
plt.show()
