import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Hyper Parameters
batch_size = 64;
no_of_epoch = 10;

# train
# Feed forward network
train_dataset = torchvision.datasets.MNIST(root = '../../data/', train = True, download = True);
test_dataset = torchvision.datasets.MNIST(root = '../../data/', train = False, download = True);

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

data_iter = iter(train_loader);
image,_ = next(data_iter);


#print(image.size()[2]*image.size()[3]);
input_size = image.size()[2]*image.size()[3];
print("size of image :" , input_size);
print("Lenght of datset:",len(train_loader.dataset));


class ffnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ffnet, self).__init__();
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes);

    def forward(self, x):
        out = self.fc1(x);
        out = self.relu(out);
        out = self.fc2(out);
        return out;

# Test modelimages
# To save compute, gradients are turned off:
mynet = ffnet(28*28, 10, 10);
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




        
         