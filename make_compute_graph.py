import torch
import torchviz

from vnet import VNet


def main():
    model = VNet(
        in_channels=1,
        out_channels=2,
        depth=5,
        wf=4,
        batch_norm=False,
        activation="prelu",
        loss="dice",
        kaiming_normal=False,
    )
    x = torch.zeros((4, 1, 128, 128, 64))
    y = model(x)
    dot = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.render(directory="./")


if __name__ == "__main__":
    main()
