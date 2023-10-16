## Pytorch basics
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms



# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print(x);
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True);
print(x)
print(x[1][2])
print(x[1][2].item())
#gives error: print(x.item())

#Create output
#Added line for push trials

# =========================  #
#   2  Update loop operation #
# =========================  #
print("#   1 Update operation      #");

# Create data and ground truth
x = torch.randn(10,3);
print(x);
y = torch.randn(10,2);

# Create neural net
linear = nn.Linear(3,2);
print("weights :", linear.weight,linear.weight.grad);
print("bias :", linear.bias, linear.bias.grad);

# Create loss and optimizer
criterion = nn.MSELoss();
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.01);

# Update step

## forward pass: prediction
pred = linear(x);
print("pred : ",pred,"end");
print(x);

## Loss calculation
loss = criterion(pred,y);
print("loss :",loss);
## calculating gradients
loss.backward();
print("weights :", linear.weight,linear.weight.grad);
print("bias :", linear.bias, linear.bias.grad);

## update with gradients
optimizer.step()

## 
pred = linear(x);
loss = criterion(pred,y);
print("loss :",loss);



# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #
xnp = np.array([[1, 2],[3,4]]);
xten = torch.from_numpy(xnp);
z = xten.numpy();


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',train=True,transform=transforms.toTensor(),download-True)

iamge,label = tran_dataset[0];
print("image :", image.size());
print("label :",label);

trainloader = torch.utils.data.Dataoader(dataset=train_dataset,batch_size=64,shuffle=True);

data_iter = iter(trainloader);

## mini-batch of image, labels
images, labels = data_iter.next();

# Actual usage of dataloader
for images, labels in trainloader:
    pass;


# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 
    
custom_loader = CustomDataset();
dataloader = torch.utils.data.Dataloader(dataset = custom_loader,batch_size=64,shuffle=True);

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #
resnet = torchvision.models.resnet18(pretrained = True);

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False;

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100);



