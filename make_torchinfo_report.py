import torchinfo

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
    torchinfo.summary(
        model,
        input_size=(4, 1, 128, 128, 64),
        depth=10,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            # "kernel_size",
            # "mult_adds",
            # "trainable",
        ],
    )


if __name__ == "__main__":
    main()
