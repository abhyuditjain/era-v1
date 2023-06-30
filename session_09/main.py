import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import Cifar10DataLoader
from model import Net
from train import Trainer
from test import Tester
from torchinfo import summary


def print_summary(model):
    batch_size = 20
    summary(
        model,
        input_size=(batch_size, 3, 32, 32),
        verbose=1,
        col_names=[
            "kernel_size",
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
            "trainable",
        ],
        row_settings=["var_names"],
    )


def main():
    is_cuda_available = torch.cuda.is_available()
    print("Is GPU available?", is_cuda_available)
    device = torch.device("cuda" if is_cuda_available else "cpu")
    cifar10 = Cifar10DataLoader(is_cuda_available=is_cuda_available)
    train_loader = cifar10.get_loader(True)
    test_loader = cifar10.get_loader(False)
    model = Net().to(device=device)

    print_summary(model)

    trainer = Trainer()
    tester = Tester()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    EPOCHS = 200

    for epoch in range(EPOCHS):
        trainer.train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        tester.test(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
