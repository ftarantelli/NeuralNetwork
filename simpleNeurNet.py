import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#torch.manual_seed(6723)

class netNeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.activation2 = nn.Tanh()
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x

def circle(x):
    r = torch.norm(x, dim=1)
    r = r.unsqueeze(1)
    return torch.exp(-(r - 3) ** 2)

loss_fun = nn.MSELoss()

rete = netNeur()

losses = []
optimizer = torch.optim.SGD(rete.parameters(), lr=0.001)

for i in range(0, 5000):
    p = 3. + torch.randn((30, 2)) * 0.4
    target = circle(p)
    result = rete(p)
    loss = loss_fun(target, result)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #rete.zero_grad()

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

p = 3. + torch.randn((1, 2)) * 0.2

rete.eval()

with torch.no_grad():
    predictions = rete(p)

print("Input p:")
print(p)
print("Predictions:")
print(predictions)
print("Circle Function Output:")
print(circle(p))



'''
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
	#print(x)
	r = torch.norm(x, dim = 1)
	r = r.unsqueeze(1)
	#print(r)
	#theta = nn.arctan(x[0]/x[1])
	#print(torch.exp(-(r-3)**2))
	#sys.exit()
	return torch.exp(-(r-3)**2)

loss_fun=nn.MSELoss()

rete=netNeur()

losses = []
for i in range(0,5000):
	p = 3. + torch.randn((10,2))*0.2
	target=circle(p)
	result=rete.forward(p)
	#print(target, result)
	#sys.exit()
	loss=loss_fun(target, result)
	losses.append(loss.item())
	loss.backward()
	#print(loss)
	for p in rete.parameters():
		p.data.sub_(0.001*p.grad)

	rete.zero_grad()

plt.plot(losses)
plt.show()

p = 3. + torch.randn((1,2))*0.2

rete.eval()

with torch.no_grad():
    predictions = rete(p)

# Print the predictions
print(p)
print(predictions)
print(circle(p))
'''
