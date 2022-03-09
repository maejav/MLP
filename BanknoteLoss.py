import torch

class BanknoteLoss(torch.nn.Module):
    def __init__(self):
        super(BanknoteLoss, self).__init__()
        self.loss = torch.nn.BCELoss()

    def forward(self, y, target):
        return(self.loss(y, target))


