import p3bo
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm
from flexs.optimizers.random import Random

protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

optimal_sequence = "MKYTKVMRYQIIKPLNAEWDELGMVLRDIQKETRAALNKTIQLCWEYQGFSADYKQIHGQYPKPKDVLGYTSMHGYAYDRLKNEFSKIASSNLSQTIKRAVDKWNSDLKEILRGDRSIPNFRKDCPIDIVKQSTKIQKCNDGYVLSLGLINREYKNELGRKNGVFDVLIKANDKTQQTILERIINGDYTYTASQIINHKNKWFINLTYQFETKETALDPNNVMGVDLGIVYPVYIAFNNSLHRYHIKGGEIERFRRQVEKRKRELLNQGKYCGDGRKGHGYATRTKSIESISDKIARFRDTCNHKYSRFIVDMALKHNCGIIQMEDLTGISKESTFLKNWTYYDLQQKIEYKAREAGIQVIKIEPQYTSQRCSKCGYIDKENRQEQATFKCIECGFKTNADYNAARNIAIPNIDKIIRKTLKMQ"


def main():
    # create a naive/mock model that simply computes the distance from the target optimum
    model = NoisyAbstractModel(LevenstheinLandscape(optimal_sequence))

    starting_sequence = p3bo.get_starting_sequence(
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

    optimizer = p3bo.P3bo([random, ga, adalead])

    # that's the method you have to implement
    optimizer.optimize()


main()
