import torch.nn as nn

class Netwerk(nn.Module):
    def __init__(self, N, D_in, H, D_out):
        super(Netwerk, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self, x):
        h = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h)
        
        return y_pred