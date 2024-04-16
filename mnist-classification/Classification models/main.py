import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
from model import LeNet5, CustomMLP, LeNet5WithDropout
import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    model.train()
    trn_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trn_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    trn_loss /= len(trn_loader)
    acc = 100 * correct / total

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    model.eval()
    tst_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tst_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            tst_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    tst_loss /= len(tst_loader)
    acc = 100 * correct / total

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate Dataset objects for training and test datasets
    train_dataset = dataset.MNIST(data_dir='../data/train')
    test_dataset = dataset.MNIST(data_dir='../data/test')

    # Instantiate DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6)

    # Instantiate the model (LeNet5 or CustomMLP)
    model_LeNet5 = LeNet5().to(device)
    model_LeNet5WihtDropout = LeNet5WithDropout().to(device)
    model_CustomMLP = CustomMLP().to(device)

    # Instantiate the optimizer and cost function for each model
    optimizer_LeNet5 = optim.SGD(model_LeNet5.parameters(), lr=0.01, momentum=0.9)
    optimizer_CustomMLP = optim.SGD(model_CustomMLP.parameters(), lr=0.01, momentum=0.9)
    optimizer_LeNet5WithDropout = optim.SGD(model_LeNet5WihtDropout.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    # Lists to store the loss and accuracy values for plotting
    train_loss_LeNet5 = []
    train_acc_LeNet5 = []
    test_loss_LeNet5 = []
    test_acc_LeNet5 = []
    train_loss_LeNet5WihtDropout = []
    train_acc_LeNet5WihtDropout = []
    test_loss_LeNet5WihtDropout = []
    test_acc_LeNet5WihtDropout = []
    train_loss_CustomMLP = []
    train_acc_CustomMLP = []
    test_loss_CustomMLP = []
    test_acc_CustomMLP = []

    # Training and testing loop
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Train and test LeNet5
        train_loss, train_acc = train(model_LeNet5, train_loader, device, criterion, optimizer_LeNet5)
        test_loss, test_acc = test(model_LeNet5, test_loader, device, criterion)
        train_loss_LeNet5.append(train_loss)
        train_acc_LeNet5.append(train_acc)
        test_loss_LeNet5.append(test_loss)
        test_acc_LeNet5.append(test_acc)
        print(f"LeNet5 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Train and test LeNet5WithDropout
        train_loss, train_acc = train(model_LeNet5WihtDropout, train_loader, device, criterion, optimizer_LeNet5WithDropout)
        test_loss, test_acc = test(model_LeNet5WihtDropout, test_loader, device, criterion)
        train_loss_LeNet5WihtDropout.append(train_loss)
        train_acc_LeNet5WihtDropout.append(train_acc)
        test_loss_LeNet5WihtDropout.append(test_loss)
        test_acc_LeNet5WihtDropout.append(test_acc)
        print(f"LeNet5WihtDropout - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Train and test CustomMLP
        train_loss, train_acc = train(model_CustomMLP, train_loader, device, criterion, optimizer_CustomMLP)
        test_loss, test_acc = test(model_CustomMLP, test_loader, device, criterion)
        train_loss_CustomMLP.append(train_loss)
        train_acc_CustomMLP.append(train_acc)
        test_loss_CustomMLP.append(test_loss)
        test_acc_CustomMLP.append(test_acc)
        print(f"CustomMLP - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


    # Plot the loss and accuracy curves for each model
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Plot the loss and accuracy curves for LeNet5
    axs[0, 0].plot(train_loss_LeNet5, label='Train Loss')
    axs[0, 0].plot(test_loss_LeNet5, label='Test Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('LeNet5 - Loss Curve')
    axs[0, 0].legend()

    axs[0, 1].plot(train_acc_LeNet5, label='Train Accuracy')
    axs[0, 1].plot(test_acc_LeNet5, label='Test Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('LeNet5 - Accuracy Curve')
    axs[0, 1].legend()

    # Plot the loss and accuracy curves for LeNet5WithDropout
    axs[1, 0].plot(train_loss_LeNet5WihtDropout, label='Train Loss')
    axs[1, 0].plot(test_loss_LeNet5WihtDropout, label='Test Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('LeNet5WithDropout - Loss Curve')
    axs[1, 0].legend()

    axs[1, 1].plot(train_acc_LeNet5WihtDropout, label='Train Accuracy')
    axs[1, 1].plot(test_acc_LeNet5WihtDropout, label='Test Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_title('LeNet5WithDropout - Accuracy Curve')
    axs[1, 1].legend()

    # Plot the loss and accuracy curves for CustomMLP
    axs[2, 0].plot(train_loss_CustomMLP, label='Train Loss')
    axs[2, 0].plot(test_loss_CustomMLP, label='Test Loss')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('Loss')
    axs[2, 0].set_title('CustomMLP - Loss Curve')
    axs[2, 0].legend()

    axs[2, 1].plot(train_acc_CustomMLP, label='Train Accuracy')
    axs[2, 1].plot(test_acc_CustomMLP, label='Test Accuracy')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('Accuracy')
    axs[2, 1].set_title('CustomMLP - Accuracy Curve')
    axs[2, 1].legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()
    exit()