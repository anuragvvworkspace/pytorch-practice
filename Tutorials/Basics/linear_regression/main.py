# Can I teacha  class how linear regression works?
# Sure!!!!

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Hyper parameters
num_epoch = 500;
learning_rate = 0.001;
# Make data
#x = np.array([[2.3], [3.4], [4.5], [5.6], [6.7],],dtype = np.float32)
#x_train = np.array([2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9,10.0],dtype = np.float32).reshape((-1,1));
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167]], dtype=np.float32)
y_train = np.random.rand(10,1)*10;
input = torch.from_numpy(x_train);
gt = torch.from_numpy(y_train.astype('float32'));

#print("x shape :", x_train.shape)
#print("y :", y_train)

linnet = nn.Linear(int(x_train.shape[1]),1);
criterion = nn.MSELoss();   
optimizer = torch.optim.SGD(linnet.parameters(),lr = learning_rate);

plt.plot(x_train, y_train, 'ro', label='Original data')
for epoch in range(num_epoch):
    pred = linnet(input);#print("size of pred:", pred.shape)
    loss = criterion(pred,gt.view(-1, 1));
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoch, loss.item()));
    print(linnet.weight,"   and   ",linnet.bias);
    #plt.plot(x_train, pred.detach().numpy())


predicted = linnet(input).detach().numpy();
pred = linnet(input);
print(predicted - pred.detach().numpy());
loss = criterion(pred,gt);
print("final loss:",loss)

plt.plot(x_train, predicted,'b-', label='Fitted line')
plt.legend()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Save the model checkpoint
torch.save(linnet.state_dict(), 'model.ckpt')

