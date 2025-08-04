import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def getDataLoaders():

    mean_grey = 0.1307
    stddev_grey = 0.3081

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_grey, std=stddev_grey)])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download = True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    # random_img = train_dataset[20][0].numpy() * stddev_grey + mean_grey

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=100)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=100)

    return train_loader, test_loader


if __name__ == "__main__":

    # plt.imshow(random_img.reshape(28,28), cmap='gray')
    # plt.show()
    pass