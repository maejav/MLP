import torch 
import torch.nn 
import torch.nn.functional as F


### set hyper parameter 
hiddenstateL1=10
output_dim=1

class BanknoteModel(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BanknoteModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hiddenstateL1)
        self.fc2 = torch.nn.Linear(hiddenstateL1, output_dim)

        ### or 
        

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

