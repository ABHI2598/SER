import sys
from typing import Tuple

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix

__author__ = 'harry-7'
__version__ = '1.1'


class Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.save_path = save_path
        self.name = name
        self.trained = False

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray,
              x_val: numpy.ndarray = None,
              y_val: numpy.ndarray = None) -> None:
        raise NotImplementedError()

    def predict(self, samples: numpy.ndarray) -> Tuple:
        results = []
        for _, sample in enumerate(samples):
            results.append(self.predict_one(sample))
        return tuple(results)

    def predict_one(self, sample) -> int:
        raise NotImplementedError()

    def restore_model(self, load_path: str = None) -> None:
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write(
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load: str) -> None:
        raise NotImplementedError()

    def save_model(self) -> None:
        raise NotImplementedError()

    def evaluate(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
        predictions = self.predict(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                                 y_true=y_test))
        print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                                    y_true=y_test))
