import torchvision.transforms as transforms

import datasets.omniglot as om


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(
        name,
        train=True,
        path=None,
        background=True,
        all=False,
        prefetch_gpu=False,
        device=None,
    ):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)), transforms.ToTensor()]
            )
            if path is None:
                return om.Omniglot(
                    "../data/omni",
                    background=background,
                    download=True,
                    train=train,
                    transform=train_transform,
                    all=all,
                    prefetch_gpu=prefetch_gpu,
                    device=device,
                )
            else:
                return om.Omniglot(
                    path,
                    download=True,
                    background=train,
                    transform=train_transform,
                    prefetch_gpu=prefetch_gpu,
                    device=device,
                )

        else:
            print("Unsupported Dataset")
            assert False
