import torch

from game_record import GameRecord
from loss import BradleyTerry_loss
from model import BradleyTerry

teams = 2
model = BradleyTerry(teams)
optimizer = torch.optim.Adam(model.parameters(), lr = 4e-3)

EPOCH = 500

games = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype = torch.int)

for i in range(EPOCH):
    outputs = model(games)
    loss = BradleyTerry_loss(outputs)
    if (i + 1) % 10 == 0:
        print(f"Step: {i} Loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for name, param in model.named_parameters():
    print(f"Parameter name: {name} | Shape: {param.shape} | Values: {param.data}")



