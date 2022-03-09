
import torch

class BanknoteOptimizer(torch.nn.Module):
    def __init__(self, parameters, lr=0.001):
        super(BanknoteOptimizer, self).__init__()
        self.optimizer = torch.optim.Adam(parameters(), lr)


    def forward():
        pass




