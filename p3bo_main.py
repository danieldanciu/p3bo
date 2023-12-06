import random

import numpy as np

import p3bo
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm
from flexs.optimizers.random import Random


def main():
    # Fix random seeds.
    random.seed(0)
    np.random.seed(0)

    protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    optimal_sequence = "MKYTKVMRYQIIKPLNAEWDELGMVLRDIQKETRAALNKTIQLCWEYQGFSADYKQIHGQYPKPKDVLGYTSMHGYAYDRLKNEFSKIASSNLSQTIKRAVDKWNSDLKEILRGDRSIPNFRKDCPIDIVKQSTKIQKCNDGYVLSLGLINREYKNELGRKNGVFDVLIKANDKTQQTILERIINGDYTYTASQIINHKNKWFINLTYQFETKETALDPNNVMGVDLGIVYPVYIAFNNSLHRYHIKGGEIERFRRQVEKRKRELLNQGKYCGDGRKGHGYATRTKSIESISDKIARFRDTCNHKYSRFIVDMALKHNCGIIQMEDLTGISKESTFLKNWTYYDLQQKIEYKAREAGIQVIKIEPQYTSQRCSKCGYIDKENRQEQATFKCIECGFKTNADYNAARNIAIPNIDKIIRKTLKMQ"

    # Create a naive/mock model that simply computes the distance from the target optimum.
    landscape = LevenstheinLandscape(optimal_sequence)
    model = NoisyAbstractModel(landscape=landscape)

    # Get a sequence 80% identical to the optimal.
    starting_sequence = p3bo.get_starting_sequence(
        alphabet=protein_alphabet, base_sequence=optimal_sequence, identity_percent=80
    )

    # Setup the explorer portfolio.
    al = Adalead(
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
    r = Random(
        model=model,
        rounds=10,
        sequences_batch_size=10,
        model_queries_per_batch=100,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
    )

    optimizer = p3bo.P3bo(
        portfolio=[r],  # , [r, ga, al],
        starting_sequence=starting_sequence,
        landscape=landscape,
        batch_size=10,
        softmax_temperature=1.0,
        decay_rate=0.9,
    )

    # That's the method you have to implement.
    optimizer.optimize(num_steps=10)


main()
