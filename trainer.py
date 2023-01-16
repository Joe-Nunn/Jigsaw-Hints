import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork
from jigsaw_piece_dataset import JigsawPieceDataset

DEFAULT_TRAINING_SIZE = 0.8  # Default proportion of dataset to use for training compared to testing
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_WEIGHT_DECAY = 0.00001
DEFAULT_TRAINING_ITERATIONS = 12


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

    def train(self, test=True, silent=False, learning_rate=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY, iterations=DEFAULT_TRAINING_ITERATIONS):
        """
        Trains the neural network using the train set of data.
        Uses binary-cross entropy loss and Adam optimisation.

        Prints loss and test accuracy each training iteration is silent is false.

        :param: test: whether to test the model while training or not
        :param: learning_rate: learning rate used by the optimiser
        :param: weight_decay: weight decay used by the optimiser
        :param: iterations: number of times the training set is passed through the neural network
        :param: silent: whether results of tests should be printed
        :return: Tuple of a list containing the loss of each iteration and the test accuracy. If test is False the second list will be empty.
        """
        criterion = nn.BCELoss()
        optimiser = torch.optim.SGD(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.network.to(self.device)  # Move model onto GPU if available

        # Initialise lists to store training results in
        losses = []
        test_accuracy = []

        # Train the network with the training set
        if not silent:
            print("Starting training on " + self.device.type)
        for i in range(iterations):
            self.network.train()  # Set network to training mode
            if not silent:
                print("Run " + str(i + 1) + ":")
            running_loss = 0
            for batch_num, data in enumerate(self.train_loader):
                images, labels = data
                # Move batch and labels to graphics card if using it
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward batch through network
                outputs = self.network(images)
                # Calculate how wrong the network output is
                loss = criterion(outputs, labels)
                # Adjust weights
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                # Update total loss for run
                running_loss += loss.item()
            #  Update average loss for iteration
            number_of_batches = len(self.train_loader)
            avg_iteration_loss = running_loss / number_of_batches
            losses.append(avg_iteration_loss)
            if test:  # Run test run
                test_results = self.test(self.test_loader)
                test_accuracy.append(test_results)
            if not silent:
                print("\t Loss: " + str(avg_iteration_loss))
            if not silent and test:
                print("\t Test accuracy: " + str(test_results))

        return losses, test_accuracy

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
                images, labels = data
                # Move batch of images to graphics card if using it
                images = images.to(self.device)
                # Forward batch through network
                predictions = self.network(images)
                # Determine how many predictions were correct
                predictions = predictions.flatten().tolist()
                labels = labels.flatten().tolist()
                for i in range(len(predictions)):
                    if round(predictions[i]) == labels[i]:
                        correct_predictions += 1

        return correct_predictions / len(loader.dataset)
