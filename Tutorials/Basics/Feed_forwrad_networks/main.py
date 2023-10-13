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
x = torch.tensor(1., autograd_required=True)
w = torch.tensor(2., autograd_required=True)
b = torch.tensor(3., autograd_required=True)


#Create output
#Added line for push trials