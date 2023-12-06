"""Implementation of the P3BO method."""
import math
import random
from typing import List

import flexs.explorer


def mutate_with_alphabet(alphabet: str, original: str) -> str:
    """Samples a new value from an alphabet."""
    return random.choice([c for c in alphabet if c != original])


def get_starting_sequence(
    alphabet: str, base_sequence: str, identity_percent: float
) -> str:
    """This function returns a sequence that is identity_percent identical to the given base sequence.

    Args:
      alphabet: The alphabet to use.
      base_sequence: The sequence to modify.
      identity_percent: The fraction of positions to keep intact.
    """

    # Maybe better to use flexs.sequence_utils generate_random_mutant,
    # but we can solve it exactly.

    n = len(base_sequence)
    num_mutations = math.ceil((1.0 - identity_percent / 100.0) * n)
    mutant = list(base_sequence)
    for pos in random.sample(population=range(n), k=num_mutations):
        mutant[pos] = mutate_with_alphabet(alphabet=alphabet, original=mutant[pos])
    return "".join(mutant)


class P3bo:
    def __init__(self, explorers: List[flexs.explorer.Explorer]):
        self.explorers = explorers

    def optimize(self):
        """
        This is the function that you need to implement (including adding the necessary parameters)

        The P3BO population will consist of the three algorithms: adalead, random and genetic (they are in the
        optimizers directory). Each of these algorithms exposes a propose_sequences() method and a fit() method.

        You don't need to implement the adaptive variant of P3BO.
        """
        pass
