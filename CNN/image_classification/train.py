import torch
import torch.nn as nn
from model import ImageClassifier
from data import getDataLoaders
from tqdm import tqdm



if __name__ == "__main__":

    model = ImageClassifier()
    train, test = getDataLoaders()

    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda(device=0)

    lossfn = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    num_epochs = 10
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        
        loop = tqdm(train, total=len(train), leave=True)

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        correct = 0
        iterations = 0
        iter_loss = 0
        
        model.train()

        for i , (inputs, labels) in enumerate(loop):

            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            output = model(inputs)
            loss = lossfn(output, labels)
            iter_loss += loss.item()

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            _, pred = torch.max(output, dim=1)
            correct += (pred == labels).sum().item()

            iterations += 1

            loop.set_postfix(train_loss=iter_loss/iterations)

        train_loss.append(iter_loss/iterations)
        train_accuracy.append(100 * (correct / (len(train)*100)))

        testing_loss = 0.0
        correct = 0
        iterations = 0

        model.eval()

        loop2 = tqdm(test, total=len(test), leave=True)

        loop2.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for i , (inputs, labels) in enumerate(loop2):

            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            output = model(inputs)
            loss = lossfn(output, labels)
            testing_loss += loss.item()

            _, pred = torch.max(output, dim=1)
            correct += (pred == labels).sum().item()

            iterations += 1

            loop2.set_postfix(test_loss=testing_loss/iterations)

        test_loss.append(testing_loss/iterations)
        test_accuracy.append(100 * (correct / (len(test)*100)))

    torch.save(model.state_dict(), "digits.pth")
    print(train_loss, train_accuracy, test_loss, test_accuracy)
