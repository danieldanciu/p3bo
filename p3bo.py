""" This is the file that contains the main P3BO method that you are supposed to implement. """
from typing import List

import flexs.explorer
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm
from flexs.optimizers.random import Random

protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

optimal_sequence = "MKYTKVMRYQIIKPLNAEWDELGMVLRDIQKETRAALNKTIQLCWEYQGFSADYKQIHGQYPKPKDVLGYTSMHGYAYDRLKNEFSKIASSNLSQTIKRAVDKWNSDLKEILRGDRSIPNFRKDCPIDIVKQSTKIQKCNDGYVLSLGLINREYKNELGRKNGVFDVLIKANDKTQQTILERIINGDYTYTASQIINHKNKWFINLTYQFETKETALDPNNVMGVDLGIVYPVYIAFNNSLHRYHIKGGEIERFRRQVEKRKRELLNQGKYCGDGRKGHGYATRTKSIESISDKIARFRDTCNHKYSRFIVDMALKHNCGIIQMEDLTGISKESTFLKNWTYYDLQQKIEYKAREAGIQVIKIEPQYTSQRCSKCGYIDKENRQEQATFKCIECGFKTNADYNAARNIAIPNIDKIIRKTLKMQ"


def get_starting_sequence(base_sequence: str, identity_percent: float) -> str:
    """This function returns a sequence that is identity_percent identical to the given base sequence"""
    pass


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


def main():
    # create a naive/mock model that simply computes the distance from the target optimum
    model = NoisyAbstractModel(LevenstheinLandscape(optimal_sequence))

    starting_sequence = get_starting_sequence(
        optimal_sequence, 80
    )  # get a sequence 80% identical to the optimal
    adalead = Adalead(
        model=model,
        rounds=10,
        sequences_batch_size=10,
        model_queries_per_batch=100,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
    )
    ga = GeneticAlgorithm(
        model=model,
        rounds=10,
        sequences_batch_size=10,
        model_queries_per_batch=100,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
        population_size=100,
        parent_selection_strategy="top-proportion",
        children_proportion=0.5,
        parent_selection_proportion=0.5,
    )
    random = Random(
        model=model,
        rounds=10,
        sequences_batch_size=10,
        model_queries_per_batch=100,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
    )

    p3bo = P3bo([random, ga, adalead])

    # that's the method you have to implement
    p3bo.optimize()


if __name__ == "__main__":
    main()
