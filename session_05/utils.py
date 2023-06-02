from math import ceil, floor, sqrt
from typing import List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary


def get_rows_cols(num: int) -> Tuple[int, int]:
    cols = floor(sqrt(num))
    rows = ceil(num / cols)

    return rows, cols


def model_summary(model, input_size):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    return summary(
        model,
        input_size=input_size,
        row_settings=["var_names"],
        col_names=["kernel_size", "input_size", "output_size", "num_params"],
    )


def plot_data(data_list: List[List[int | float]], titles: List[str]):
    assert len(data_list) == len(
        titles
    ), "length of datalist should be equal to length of title list"

    rows, cols = get_rows_cols(len(data_list))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j

            if idx >= len(data_list):
                break

            ax = axs[i, j] if len(axs.shape) > 1 else axs[max(i, j)]

            ax.plot(data_list[idx])  # type: ignore
            ax.set_title(titles[idx])


def visualize_data(loader, num_figures: int = 12, label: str = ""):
    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()
    fig.suptitle(label)

    rows, cols = get_rows_cols(num_figures)

    for i in range(num_figures):
        plt.subplot(rows, cols, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.CenterCrop(22),
                ],
                p=0.1,
            ),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15.0, 15.0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def get_dataloaders(batch_size: int = 512):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }

    train_data = datasets.MNIST(
        "../data", train=True, download=True, transform=get_train_transforms()
    )
    test_data = datasets.MNIST(
        "../data", train=False, download=True, transform=get_test_transforms()
    )

    train_loader = data.DataLoader(train_data, **kwargs)
    test_loader = data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader


def get_correct_pred_count(pred, targets):
    return pred.argmax(dim=1).eq(targets).sum().item()
    # return pred.eq(targets.view_as(pred)).sum().item()


class Trainer:
    def __init__(self) -> None:
        self.losses = []
        self.accuracies = []

    def train(self, model, device, train_loader, optimizer, criterion, epoch=1):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += get_correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train set: LR = {optimizer.param_groups[0]['lr']} | Loss = {loss.item():3.2f} | Batch = {batch_idx} | Accuracy = {100*correct/processed:0.2f} |"
            )

        self.accuracies.append(100 * correct / processed)
        self.losses.append(train_loss / len(train_loader))


class Tester:
    def __init__(self) -> None:
        self.losses = []
        self.accuracies = []

    def test(self, model, device, test_loader, criterion, epoch=1):
        model.eval()
        # pbar = tqdm(test_loader)

        test_loss = 0
        correct = 0
        processed = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                correct += get_correct_pred_count(output, target)
                processed += len(data)

        test_loss /= len(test_loader.dataset)

        self.accuracies.append(100.0 * correct / len(test_loader.dataset))
        self.losses.append(test_loss)

        print(
            "Test set: Average loss = {:.4f} | Accuracy = {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


def fit_model(model, train_loader, test_loader, num_epochs: int = 20):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    trainer, tester = Trainer(), Tester()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        trainer.train(model, device, train_loader, optimizer, criterion, epoch)
        tester.test(model, device, test_loader, criterion, epoch)
        scheduler.step()

    return collect_results(trainer, tester)


def collect_results(trainer: Trainer, tester: Tester):
    return {
        "train_losses": trainer.losses,
        "train_accuracies": trainer.accuracies,
        "test_losses": tester.losses,
        "test_accuracies": tester.accuracies,
    }
