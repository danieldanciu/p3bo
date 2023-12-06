import argparse
import datetime as dt
import random

import numpy as np
import tensorboardX

import p3bo
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm
from flexs.optimizers.random import Random


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3BO")

    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument("--run_name", type=str, default=None, help="Name for the run.")
    parser.add_argument(
        "--sequences_batch_size", type=int, default=15, help="Sequences batch size."
    )
    parser.add_argument(
        "--model_queries_per_batch",
        type=int,
        default=150,
        help="Model queries per batch.",
    )
    parser.add_argument(
        "--softmax_temperature", type=float, default=1.0, help="Softmax temperature."
    )
    parser.add_argument("--decay_rate", type=float, default=0.9, help="Decay rate.")
    parser.add_argument(
        "--rounds", type=int, default=10, help="Maximum number of steps."
    )
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size.")
    parser.add_argument(
        "--signal_strength",
        type=float,
        default=1.0,
        help="Signal strength for the noisy model.",
    )

    # Algorithm specific flags.
    parser.add_argument(
        "--ga_population_size",
        type=int,
        default=100,
        help="Population size for the genetic algorithm.",
    )
    parser.add_argument(
        "--ga_parent_selection_strategy",
        type=str,
        default="top-proportion",
        help="Parent selection strategy for the genetic algorithm.",
    )
    parser.add_argument(
        "--ga_children_proportion",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ga_parent_selection_proportion",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Fix random seeds.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Problem statement.
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
        rounds=args.rounds,
        sequences_batch_size=args.sequences_batch_size,
        model_queries_per_batch=args.model_queries_per_batch,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
    )
    ga = GeneticAlgorithm(
        model=model,
        rounds=args.rounds,
        sequences_batch_size=args.sequences_batch_size,
        model_queries_per_batch=args.model_queries_per_batch,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
        population_size=args.ga_population_size,
        parent_selection_strategy=args.ga_parent_selection_strategy,
        children_proportion=args.ga_children_proportion,
        parent_selection_proportion=args.ga_parent_selection_proportion,
        seed=args.random_seed,
    )
    r = Random(
        model=model,
        rounds=args.rounds,
        sequences_batch_size=args.sequences_batch_size,
        model_queries_per_batch=args.model_queries_per_batch,
        starting_sequence=starting_sequence,
        alphabet=protein_alphabet,
        seed=args.random_seed,
    )

    optimizer = p3bo.P3bo(
        portfolio=[r],  # , [r, ga, al],
        starting_sequence=starting_sequence,
        model=model,
        landscape=landscape,
        batch_size=args.batch_size,
        softmax_temperature=args.softmax_temperature,
        decay_rate=args.decay_rate,
    )

    # TODO: add more arguments.
    parameters = ",".join(
        [
            f"b={args.batch_size}",
            f"t={args.softmax_temperature}",
            f"dr={args.decay_rate}",
            f"ss={args.signal_strength}",
        ]
    )
    run_name = (
        f"{args.run_name or dt.datetime.now().strftime('%b%d_%H-%M-%S')} {parameters}"
    )

    with tensorboardX.SummaryWriter(logdir=f"runs/{run_name}") as summary_writer:
        optimizer.optimize(num_steps=args.rounds, summary_writer=summary_writer)


main()
