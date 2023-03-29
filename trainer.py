import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork
from jigsaw_piece_dataset import JigsawPieceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEFAULT_TRAINING_SIZE = 0.9  # Default proportion of dataset to use for training compared to testing
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_TRAINING_EPOCHS = 16
BATCHES_PER_EPOCH = 128  # 2048 samples per epoch


class Trainer:
    """
    Trains and tests a neural network to identify if an image of a jigsaw piece comes from a section of the base image
    """

    def __init__(self, dataset_path, csv_name, batch_size=DEFAULT_BATCH_SIZE, training_size=DEFAULT_TRAINING_SIZE):
        self.network = NeuralNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.batch_size = batch_size
        self.training_size = training_size
        self.train_loader, self.test_loader = self.__prepare_dataset(dataset_path, csv_name)

    def __prepare_dataset(self, folder_path, csv_name):
        """
        Loads the dataset splitting it into training and testing sets
        Returns data loaders for the training and testing set
        """
        dataset = JigsawPieceDataset(csv_name, folder_path)

        # Calculate number of images to be used for training and testing
        n_train_images = round(dataset.__len__() * self.training_size)
        n_test_images = dataset.__len__() - n_train_images

        # Split the dataset into training and testing set
        train_set, test_set = torch.utils.data.random_split(dataset, [n_train_images, n_test_images])

        # Create data loaders from split dataset
        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self, test=True, silent=False, learning_rate=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY, epochs=DEFAULT_TRAINING_EPOCHS):
        """
        Trains the neural network using the train set of data.
        Uses binary-cross entropy loss and SGD optimisation.

        Prints loss and test accuracy each training epoch if silent is false.

        :param: test: whether to test the model while training or not
        :param: learning_rate: learning rate used by the optimiser
        :param: weight_decay: weight decay used by the optimiser
        :param: epochs: number of times batches from the training set are passed through the neural network
        :param: silent: whether results of tests should be printed
        :return: Tuple of a list containing the loss of each epoch and the test accuracy. If test is False the test accuracy lists will be empty
        """
        criterion = nn.BCELoss()
        optimiser = torch.optim.SGD(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Every time loss has decreased for four epochs learning rate is reduced by half
        schedular = ReduceLROnPlateau(optimiser, patience=4, verbose=not silent, factor=0.5)

        self.network.to(self.device)  # Move model onto GPU if available

        # Initialise lists to store training results in
        losses = []
        test_set_accuracy = []
        train_set_accuracy = []

        # Train the network with the training set
        if not silent:
            print("Starting training on " + self.device.type)
        for i in range(epochs):
            self.network.train()  # Set network to training mode
            if not silent:
                print("Epoch " + str(i + 1) + ":")
            running_loss = 0
            for batch_num, data in enumerate(self.train_loader):
                pieces, base_sections, labels = data
                # Move batch and labels to graphics card if using it
                pieces = pieces.to(self.device)
                base_sections = base_sections.to(self.device)
                labels = labels.to(self.device)
                # Forward batch through network
                outputs = self.network(pieces, base_sections)
                # Calculate how wrong the network output is
                loss = criterion(outputs, labels)
                # Adjust weights
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                # Update total loss for run
                running_loss += loss.item()
                if batch_num >= BATCHES_PER_EPOCH - 1:  # End epoch after required number of random batches
                    break

            #  Update average loss for epoch
            avg_epoch_loss = running_loss / BATCHES_PER_EPOCH
            schedular.step(avg_epoch_loss)
            losses.append(avg_epoch_loss)
            if test:
                # Run test run on training set
                test_set_results = self.test(self.test_loader)
                test_set_accuracy.append(test_set_results)
                # Run test run on test set
                train_set_results = self.test(self.train_loader)
                train_set_accuracy.append(train_set_results)
                if test_set_results == max(test_set_accuracy):
                    self.save_model("best_model.pt")
                    if not silent:
                        print("\t New best model saved")
            # Print results of epoch
            if not silent:
                print("\t Loss: " + str(avg_epoch_loss))
            if not silent and test:
                print("\t Test set accuracy: " + str(test_set_results))
                print("\t Train set accuracy: " + str(train_set_results))

        self.save_model("final_model.pt")
        return losses, test_set_accuracy, train_set_accuracy

    def test(self, loader=None):
        """
        Tests the accuracy of the neural network

        :param: loader to use for testing
        :return: accuracy. Number of correct predictions / all predictions
        """
        if loader is None:
            loader = self.test_loader

        self.network.eval()  # Set network to evaluation mode
        self.network.to(self.device)  # Move model onto GPU if available

        correct_predictions = 0

        with torch.no_grad():  # Gradient isn't needed in testing, not using it increases performance
            for batch_num, data in enumerate(loader):
                pieces, base_sections, labels = data
                # Move batch of images to graphics card if using it
                pieces = pieces.to(self.device)
                base_sections = base_sections.to(self.device)
                # Forward batch through network
                predictions = self.network(pieces, base_sections)
                # Determine how many predictions were correct
                predictions = predictions.flatten().tolist()
                labels = labels.flatten().tolist()
                for i in range(len(predictions)):
                    if round(predictions[i]) == labels[i]:
                        correct_predictions += 1

        return correct_predictions / len(loader.dataset)

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
