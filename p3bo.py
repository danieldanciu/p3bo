"""Implementation of the P3BO method."""
import dataclasses
import math
import random
from typing import Generator, List

import numpy as np
import pandas as pd
import scipy.special
import tqdm

import flexs.explorer
import flexs.landscape


@dataclasses.dataclass
class Measurement:
    """A measured sequence with ground truth score."""

    round: int
    sequence: str
    model_score: float
    true_score: float


@dataclasses.dataclass
class Candidate:
    """A sequence proposed by an explorer."""

    sequence: str
    model_score: float


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


def sample_from_explorer(
    explorer: flexs.explorer.Explorer, measured_sequences: pd.DataFrame
) -> Generator[Candidate, None, None]:
    # TODO: Is this correct?
    # Maybe bound the number of iterations.
    while True:
        for sequence, score in zip(
            *explorer.propose_sequences(measured_sequences=measured_sequences)
        ):
            yield Candidate(sequence=sequence, model_score=score)


class P3bo:
    def __init__(
        self,
        starting_sequence: str,
        landscape: flexs.landscape.Landscape,
        portfolio: List[flexs.explorer.Explorer],
        batch_size: int,
        softmax_temperature: float,
        decay_rate: float,
    ):
        # Configuration.
        self.landscape = landscape
        self.portfolio_size = len(portfolio)
        assert self.portfolio_size > 0
        self.portfolio = portfolio
        self.softmax_temperature = softmax_temperature
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.starting_sequence = starting_sequence

        # Current state.
        self.current_step = 0
        self.sampling_weights = (
            np.ones(shape=(self.portfolio_size,)) / self.portfolio_size
        )
        self.credit_score = np.zeros(shape=(self.portfolio_size,))

        # Initialize.
        self.measured_sequences = pd.DataFrame(
            [
                Measurement(
                    round=self.current_step,
                    sequence=self.starting_sequence,
                    model_score=np.nan,
                    true_score=landscape.get_fitness([self.starting_sequence]).item(),
                )
            ]
        )
        self.all_sequences = set(self.starting_sequence)

    def reward(
        self,
        fmax: float,
        batch_measurements: pd.DataFrame,
        proposed_sequences: List[Candidate],
    ) -> float:
        """Computes the rewards for an explorer after the measurements have been taken.

        Args:
            batch_measurements: The sequences and ground truth scores for the current batch.
            proposed_sequences: The sequences proposed by an explorer in this batch.

        Returns:
            The raw (undecayed) reward.
        """
        return (
            max(
                batch_measurements.true_score[candidate.sequence]
                for candidate in proposed_sequences
            )
            / fmax
            - 1.0
        )

    def step(self):
        """Performs one step of the P3BO optimizer."""
        self.current_step += 1

        batch_sequences = {}
        batch_sequences_by_explorer = [[] for _ in range(self.portfolio_size)]
        explorer_samplers = [
            sample_from_explorer(
                explorer=explorer, measured_sequences=self.measured_sequences
            )
            for explorer in self.portfolio
        ]

        # Generate a batch of sequences.
        while len(batch_sequences) < self.batch_size:
            sampler_id = np.random.choice(
                range(self.portfolio_size), p=self.sampling_weights
            )

            candidate = next(explorer_samplers[sampler_id])
            if candidate.sequence not in self.all_sequences:
                self.all_sequences.add(candidate.sequence)
                batch_sequences[candidate.sequence] = candidate

            # Record proposed sample for the explorer.
            if candidate.sequence in batch_sequences:
                batch_sequences_by_explorer[sampler_id].append(candidate)

        # Measure the true score.
        batch_candidates = list(batch_sequences.values())
        batch_true_score = self.landscape.get_fitness(
            [candidate.sequence for candidate in batch_candidates]
        )

        batch_measured_sequences = pd.DataFrame(
            [
                Measurement(
                    round=self.current_step,
                    sequence=candidate.sequence,
                    model_score=candidate.model_score,
                    true_score=true_score,
                )
                for candidate, true_score in zip(batch_candidates, batch_true_score)
            ]
        )

        # Record the sequences measured in this batch.
        self.measured_sequences = pd.concat(
            [self.measured_sequences, batch_measured_sequences], ignore_index=True
        )

        # Compute rewards.
        batch_measured_sequences = batch_measured_sequences.set_index(
            batch_measured_sequences.sequence
        )
        fmax = self.measured_sequences.true_score.max()
        rewards = np.array(
            [
                self.reward(
                    fmax=fmax,
                    batch_measurements=batch_measured_sequences,
                    proposed_sequences=candidates,
                )
                for candidates in batch_sequences_by_explorer
            ]
        )

        # Compute the credit scores: the decayed rewards.
        self.credit_score = rewards + self.decay_rate * self.credit_score

        # Adjsut the sampling weights.
        self.sampling_weights = scipy.special.softmax(self.credit_score)

        # Retrain the explorers on all available data.
        for explorer in self.portfolio:
            explorer.fit(
                sequences=self.measured_sequences.sequence.to_numpy(),
                fitness_values=self.measured_sequences.true_score.to_numpy(),
            )

    def optimize(self, num_steps: int):
        """Runs the optimizer for a number of steps.

        This is the function that you need to implement (including adding the necessary parameters)

        The P3BO population will consist of the three algorithms: adalead, random and genetic (they are in the
        optimizers directory). Each of these algorithms exposes a propose_sequences() method and a fit() method.

        You don't need to implement the adaptive variant of P3BO.
        """
        for _ in (
            progress := tqdm.tqdm(range(num_steps), leave=False, desc="Optimizing ...")
        ):
            progress.set_description(f"Step {self.current_step}")
            self.step()
