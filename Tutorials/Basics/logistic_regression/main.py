import numpy as np
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms

#Hyper parameters
batch_size = 100;
no_of_classes = 10;
learning_rate = 0.01;
no_of_epoch = 5;


train_data = torchvision.datasets.MNIST(root='../../data/',train=True, transform=transforms.ToTensor(),download = True);
test_data = torchvision.datasets.MNIST(root='../../data/',train=False, transform=transforms.ToTensor(),download = True);

train_loader = torch.utils.data.DataLoader(dataset = train_data,shuffle = True, batch_size=batch_size);
test_loader = torch.utils.data.DataLoader(dataset = test_data,shuffle = True, batch_size=batch_size);

data_iter = iter(train_loader);
image,_ = next(data_iter);
#print(image.size()[2]*image.size()[3]);
input_size = image.size()[2]*image.size()[3];
mynet = nn.Linear(input_size, no_of_classes);

criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.SGD(mynet.parameters(), lr = learning_rate);


# Test modelimages
# To save compute, gradients are turned off:
with torch.no_grad():
    correct = 0;
    total = 0;
    for images,labels in test_loader:
        pred = mynet(images.reshape(-1,input_size));
        #print(pred[0]);
        #print(labels.size());
        _,predicted = torch.max(pred.data,1);

        total+=labels.size(0);
        correct+=(predicted == labels).sum().item();
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# Train model
for epoch in range(no_of_epoch):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,input_size);
        pred = mynet(images);
        loss = criterion(pred,labels);#print(loss,i);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        #print(images.size(),i);

# Test modelimages
# To save compute, gradients are turned off:
with torch.no_grad():
    correct = 0;
    total = 0;
    for images,labels in test_loader:
        pred = mynet(images.reshape(-1,input_size));
        #print(pred[0]);
        #print(labels.size());
        _,predicted = torch.max(pred.data,1);

        total+=labels.size(0);
        correct+=(predicted == labels).sum().item();
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


    



