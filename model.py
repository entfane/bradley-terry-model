import torch
import torch.nn as nn

class BradleyTerry(nn.Module):

    def __init__(self, teams):
        super(BradleyTerry, self).__init__()
        self.strengths = nn.Parameter(torch.ones(teams))

    def forward(self, x):
        winners = x[:, 0]
        losers = x[:, 1]
        probs = torch.exp(self.strengths[winners]) / (torch.exp(self.strengths[winners]) + torch.exp(self.strengths[losers]))
        return probs



