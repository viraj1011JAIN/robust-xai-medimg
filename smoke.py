import timm
import torch
import torchvision


def main():
    print("torch      =", torch.__version__, "| cuda =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("torchvision =", torchvision.__version__)
    print("timm       =", timm.__version__)


if __name__ == "__main__":
    main()
