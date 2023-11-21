import pandas as pd
import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from model import GRUClassif
from skorch import NeuralNetClassifier
import skorch
from data_processing import save_data

def load_data(file_path):
    # Load processed data from CSV file
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    # Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    # Take 80% from the start as training set, 20% from the end as validation set
    # TRAINING DATA
    train_1 = df[:int(len(df))-442]
    # Split the training set to training and validation sets
    train = train_1[:int(len(df)*0.8)]
    X_train = torch.tensor(train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:, -1].values, dtype=torch.int64)
    # Save training data
    save_data(train, './data/train.csv')
    # VALIDATION DATA
    val = train_1[int(len(train_1)*0.8):]
    X_val = torch.tensor(val.iloc[:, :-1].values, dtype=torch.float32)
    y_val = torch.tensor(val.iloc[:, -1].values, dtype=torch.int64)
    # Save validation data
    save_data(val, './data/val.csv')
    # TEST DATA
    test = df[int(len(df))-442:].iloc[:, :-1]
    # Save testing data
    save_data(test, './data/test.csv')
    return X_train, X_val, y_train, y_val

def validate(model, X_val, y_val, batch_size, loss_fn, epoch):
    # Metrics
    val_loss = []
    val_acc = []
    batches_per_epoch = math.ceil(len(X_val) / batch_size)

    for i in range(batches_per_epoch):
        start = i * batch_size
        if (start+batch_size) > len(X_val):
            # take a batch
            Xbatch = X_val[start:]
            ybatch = y_val[start:]
        else:
            # take a batch
            Xbatch = X_val[start:start+batch_size]
            ybatch = y_val[start:start+batch_size]
        # forward pass
        logits = model(Xbatch)
        y_pred = torch.argmax(logits, dim=1)
        loss = loss_fn(logits, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
        # store metrics
        val_loss.append(float(loss))
        val_acc.append(float(acc))

    # Average losses and accuracies
    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_acc) / len(val_acc)

    # print progress
    print(f"Validation epoch {epoch} loss {val_loss} accuracy {val_acc}")

    return val_loss, val_acc

def grid_search(X, y):
    # Define search space
    space = dict()
    space['module__num_layers'] = [2, 3, 4]
    space['module__d_model'] = [64, 128, 256, 512]

    # Define model for grid search using skorch
    model = NeuralNetClassifier(
        module=GRUClassif,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        module__input_dim=74,
        module__output_dim=9,
        module__dropout=0.1,
        batch_size=64,
        max_epochs=100,
        train_split=skorch.dataset.ValidSplit(cv=5, stratified=False, random_state=None),
    )

    # Define search
    search = GridSearchCV(model, space, error_score='raise', n_jobs=-1, cv=3)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    return result.best_params_

def plot_loss(n_epochs, loss, val_loss):
    # Plot the loss metrics
    epochs = range(1, n_epochs+1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, n_epochs, 3))
    plt.legend(loc='best')
    plt.savefig('data/train_val_loss.png')

def train_model(X_train, y_train, X_val, y_val, hyperparams={'module__d_model': 64, 
    'module__num_layers': 2}):
    ''' Initialize the model and train it'''
    model = GRUClassif(input_dim=74, output_dim=9, d_model=hyperparams['module__d_model'], 
        num_layers=hyperparams['module__num_layers'], dropout=0.1)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100  # number of epochs to run, smaller value, like 50 would be good too
    batch_size = 64  # size of each batch
    batches_per_epoch = math.ceil(len(X_train) / batch_size)

    # collect metrics
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # Training and validation
    for epoch in range(n_epochs):
        train_l = []
        train_a = []
        # Training loop
        for i in range(batches_per_epoch):
            start = i * batch_size
            if (start+batch_size) > len(X_val):
                # take a batch
                Xbatch = X_train[start:]
                ybatch = y_train[start:]
            else:
                # take a batch
                Xbatch = X_train[start:start+batch_size]
                ybatch = y_train[start:start+batch_size]
            # forward pass
            logits = model(Xbatch)
            y_pred = torch.argmax(logits, dim=1)
            loss = loss_function(logits, ybatch)
            acc = (y_pred.round() == ybatch).float().mean()
            # store metrics
            train_l.append(float(loss))
            train_a.append(float(acc))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        # Average losses and accuracies
        train_loss.append((sum(train_l)/len(train_l)))
        train_acc.append((sum(train_a)/len(train_a)))
        # print progress
        print(f"Training epoch {epoch} loss {(sum(train_l)/len(train_l))} accuracy {(sum(train_a)/len(train_a))}")
        # Validation
        v_loss, v_acc = validate(model, X_val, y_val, batch_size, loss_function, epoch)
        val_loss.append(v_loss)
        val_acc.append(v_acc)

    # Plot the training and validation loss
    plot_loss(n_epochs, train_loss, val_loss)
    return model

def save_model(model, model_path):
    # Save your trained model
    torch.save(model, model_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--tuning_mode', 
        action="store_true", 
        help='Turn on hyperparameter tuning'
    )
    return parser.parse_args()

def main(input_file, model_file, tuning_mode):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    if tuning_mode:
        # Do hyperparameter tuning
        params = grid_search(X_train, y_train)
        # Train model with found parameters
        model = train_model(X_train, y_train, X_val, y_val, params)
    else:
        model = train_model(X_train, y_train, X_val, y_val)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.tuning_mode)
