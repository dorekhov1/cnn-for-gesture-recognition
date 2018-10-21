import torch
import numpy as np
from model import Net

data = torch.Tensor(np.load("data/normalized_test.npy").swapaxes(1, 2))

model = Net(10, 6)
model.load_state_dict(torch.load("models/model", map_location='cpu'))

predictions = model(data)
predictions = predictions.argmax(dim=1)
np.savetxt('predictions.txt', predictions)
