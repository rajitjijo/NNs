from torchvision import datasets, transforms
import torch
from model import ImageClassifier
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    
    model = ImageClassifier()
    model.load_state_dict(torch.load("digits.pth"))

    mean_grey = 0.1307
    stddev_grey = 0.3081

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean_grey,), std=(stddev_grey,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)
    model.eval()

    image, label = test_dataset[200]

    # add batch dimension (1, 1, 28, 28)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, dim=1)    

    print(f"True label: {label}, Predicted: {pred.item()}")

    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.show()