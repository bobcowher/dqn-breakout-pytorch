# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from breakout import *
from utils import *
from agent import *
from model import *
import torch
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device)

model = AtariNet(nb_actions=4)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=1.0,
              min_epsilon=0.1,
              nb_warmup=5000, # originally 10000
              nb_actions=4,
              memory_capacity=20000,
              batch_size=32)

agent.train(env=environment, epochs=100000)

test_environment = DQNBreakout(device=device, render_mode='human')

agent.test(env=test_environment)







# Display the image
# display_observation_image(state)

