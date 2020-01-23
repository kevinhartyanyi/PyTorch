import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import imageio

D_in = 1
H = 10
D_out = 1


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

# Linear
y = x + torch.unsqueeze(torch.rand(100) - 0.5, dim=1)

# Polynomial
y = x**2 + torch.unsqueeze(torch.rand(100), dim=1)


print(x.size())
print(y.size())

plt.plot(x,y,'ro')
plt.show()

x,y = Variable(x), Variable(y)

class Regression(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(Regression, self).__init__()
        self.lin = nn.Linear(input_dim, hidden)
        self.lin2 = nn.Linear(hidden, output_dim)
    def forward(self, x):
        x = F.relu(self.lin(x))
        x = self.lin2(x)
        return x

model = Regression(D_in, H, D_out)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

gif_img = []
fig, ax = plt.subplots(figsize=(12,7))

for i in tqdm.tqdm(range(200)):
    
    predict = model(x)
    loss = loss_fn(predict, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    #print(loss.item())

    plt.cla() # Clear axes
    ax.set_title("Training")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-1.05, 1.5)
    ax.set_ylim(-0.25, 1.25)
    ax.scatter(x.data.numpy(), y.data.numpy(), color="orange")
    ax.plot(x.data.numpy(), predict.data.numpy(), 'g-', lw=3)
    ax.text(1.0, 0.1, 'Step = %d' % i, fontdict={'size': 24, 'color':  'red'})
    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color':  'red'})

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    gif_img.append(image)

imageio.mimsave('./linear.gif', gif_img, fps=10)
