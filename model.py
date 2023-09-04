import torch
import torch.nn as nn


class AtariNet(nn.Module):

    def __init__(self, nb_actions=6):

        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 512)

        self.output = nn.Linear(512, nb_actions)

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.output(x)
        return x


def save_the_model(model, weights_filename=None):
    torch.save(model.state_dict(), weights_filename)


def build_the_model(weights_filename=None, test_run=False, display_summary=False, nb_actions=6):
    model = AtariNet(nb_actions)

    if weights_filename is not None:
        try:
            model.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")

    if test_run:
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters())

    if display_summary:
        print(model)

    return model

