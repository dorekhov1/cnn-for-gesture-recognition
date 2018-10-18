'''
    The entry into your code. This file should include a training function and an evaluation function.
'''

import torch

from model import Net
from util import *
import pandas as pd
import time

import plot_train
import string
from sklearn.preprocessing import label_binarize


def evaluate(net, loader, criterion, needs_o_h):

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # device_type = "cpu"
    device = torch.device(device_type)

    total_loss = 0.0
    total_err = 0.0

    for i, data in enumerate(loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = torch.Tensor(label_binarize(labels, classes=range(0, 26))).to(device) if needs_o_h else labels.to(device)
        outputs = net(inputs).to(device)
        try:
            loss = criterion(outputs, labels, reduction="elementwise_mean")
        except RuntimeError:
            loss = criterion(outputs, labels.float().to(device))

        if needs_o_h:
            total_err += torch.sum(labels.argmax(dim=1) != outputs.argmax(dim=1)).item()
        else:
            total_err += torch.sum(labels != outputs.argmax(dim=1)).item()

        total_loss += loss.item()

    err = float(total_err) / len(loader.dataset)
    loss = float(total_loss) / len(loader.dataset)

    return err, loss


def main():

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # device_type = "cpu"
    device = torch.device(device_type)

    print("Device type: %s"%device_type)

    torch.manual_seed(1000)

    config, learning_rate, batch_size, num_epochs, loss, optim, model, split, s = load_config('configuration.json')

    split_data(split, s)

    train_loader, val_loader = get_data_loader(batch_size)

    net = Net(model, train_loader.dataset.X.shape[1]).to(device)

    criterion, needs_o_h = choose_loss(loss)
    optimizer = choose_optimizer(optim)(net.parameters(), lr=learning_rate, weight_decay=0.001)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            inputs = inputs.to(device)

            labels = torch.Tensor(label_binarize(labels, classes=range(0, 26))).to(device) \
                if needs_o_h else labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float()).to(device)

            try:
                loss = criterion(outputs, labels, reduction="elementwise_mean")
            except RuntimeError:
                loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()

            if needs_o_h:
                total_train_err += torch.sum(labels.argmax(dim=1) != outputs.argmax(dim=1)).item()
            else:
                total_train_err += torch.sum(labels != outputs.argmax(dim=1)).item()

            total_train_loss += loss.item()

        # if epoch % 10 == 1:
        learning_rate = 0.95 * learning_rate
        optimizer = choose_optimizer(optim)(net.parameters(), lr=learning_rate)

        train_err[epoch] = float(total_train_err) / len(train_loader.dataset)
        train_loss[epoch] = float(total_train_loss) / len(train_loader.dataset)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion, needs_o_h)

        if val_err[epoch] < 0.12:
            torch.save(net.state_dict(), "models/" + str(val_err[epoch]))

        print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}"
              .format(epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    model_name = get_model_name(config)
    model_path = "results/"+model_name

    model_path = make_model_directory(model_path)

    torch.save(net.state_dict(), model_path+"/"+model_name)

    epochs = np.arange(1, num_epochs + 1)

    df = pd.DataFrame({"epoch": epochs, "train_err": train_err})
    df.to_csv("{}/train_err_{}.csv".format(model_path, model_name), index=False)

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("{}/train_loss_{}.csv".format(model_path, model_name), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_err": val_err})
    df.to_csv("{}/val_err_{}.csv".format(model_path, model_name), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("{}/val_loss_{}.csv".format(model_path, model_name), index=False)

    plot_train.plot_train(model_path, model_name)


if __name__ == '__main__':
    main()
