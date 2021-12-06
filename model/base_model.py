from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self):
        """ Load input and output data as training and test datasets"""
        pass

    @abstractmethod
    def build(self):
        """ Create the model, set loss metrics and optimization algorithm and parameters"""
        pass

    @abstractmethod
    def train_regularizer(self):
        """ Train the network that acts as a regularization functional """
        pass
