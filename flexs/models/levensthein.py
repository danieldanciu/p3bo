from typing import List, Any

import editdistance
import flexs.landscape
import numpy as np


class LevenstheinLandscape(flexs.landscape.Landscape):
    """ A simple landscape that returns the Levensthein distance from the optimum sequence as the fitness value """

    def __init__(self, target_sequence):
        super().__init__(target_sequence)
        self.target_sequence = target_sequence

    def _fitness_function(self, sequences: flexs.model.SEQUENCES_TYPE) -> np.ndarray:
        result = []
        for seq in sequences:
            result.append(editdistance.eval(seq, self.target_sequence))
        return np.array(result)

    def train(self, sequences: flexs.model.SEQUENCES_TYPE, labels: List[Any]):
        """ The Levensthein model cannot be trained """
        pass
