import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self):
        return self.embedding_net


class QuadrupletNet(nn.Module):
    def __init__(self, embedding_net):
        super(QuadrupletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3, x4):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        output4 = self.embedding_net(x4)
        return output1, output2, output3, output4

    def get_embedding(self, x):
        return self.embedding_net(x)