""" This is the file that contains the main P3BO method that you are supposed to implement. """
import random
from typing import List, Dict, Set
import numpy as np
import pandas as pd

import flexs.explorer
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.random import Random
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm

protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

optimal_sequence = "MKYTKVMRYQIIKPLNAEWDELGMVLRDIQKETRAALNKTIQLCWEYQGFSADYKQIHGQYPKPKDVLGYTSMHGYAYDRLKNEFSKIASSNLSQTIKRAVDKWNSDLKEILRGDRSIPNFRKDCPIDIVKQSTKIQKCNDGYVLSLGLINREYKNELGRKNGVFDVLIKANDKTQQTILERIINGDYTYTASQIINHKNKWFINLTYQFETKETALDPNNVMGVDLGIVYPVYIAFNNSLHRYHIKGGEIERFRRQVEKRKRELLNQGKYCGDGRKGHGYATRTKSIESISDKIARFRDTCNHKYSRFIVDMALKHNCGIIQMEDLTGISKESTFLKNWTYYDLQQKIEYKAREAGIQVIKIEPQYTSQRCSKCGYIDKENRQEQATFKCIECGFKTNADYNAARNIAIPNIDKIIRKTLKMQ"


def get_starting_sequence(base_sequence: str, identity_percent: float) -> str:
    """This function returns a sequence that is identity_percent identical to the given base sequence"""
    sequence: List[str] = list(base_sequence)
    num_mutations: int = round(len(base_sequence) * (1.0 - identity_percent))
    indices = set()
    while len(indices) < num_mutations:
        # Pick the next index to replace at random. Make sure it hasn't already been mutated.
        index = random.randint(0, len(sequence) - 1)
        if index in indices:
            continue

        # Pick a mutation at random from the protein alphabet and make sure it's a change by
        # removing the current aminoacid from the alphabet.
        sequence[index] = random.choice(protein_alphabet.replace(sequence[index], ""))
        indices.add(index)
    return "".join(sequence)


class P3bo:
    def __init__(
        self,
        explorers: List[flexs.explorer.Explorer],
        landscape: flexs.landscape.Landscape,
        sequences_batch_size: int,
    ):
        self.explorers = explorers
        self.landscape = landscape
        self.sequences_batch_size = sequences_batch_size

    def _initial_sequence_data(self, explorer: flexs.explorer.Explorer) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sequence": explorer.starting_sequence,
                "model_score": np.nan,
                "true_score": self.landscape.get_fitness([explorer.starting_sequence]),
                "round": 0,
            }
        )


    def _get_rewards(self, fmax: float, sequences_to_test: List[str], true_scores: np.ndarray, explorer_sequences: Dict[str, Set[str]]):
        """Returns a per explorer reward."""            
        maxvalue = {}  # explorer => max score
        for i, s in enumerate(sequences_to_test):
            for e, sequences in explorer_sequences.items():
                if s in sequences:
                    maxvalue[e] = max(maxvalue.get(e, 0.0), true_scores[i])
        reward = {}  # explorer => reward
        for explorer in self.explorers:
          reward[explorer.name] = (maxvalue.get(explorer.name, 0.0) - fmax) / fmax
        return reward
    
    def _decayed_rewards(self, rewards_history: List[Dict[str, float]], gamma: float = 0.9):
        """Returns the decayed rewards given the reward history (sorted from oldest to latest)."""
        decayed_rewards = {}
        for i, rewards in enumerate(rewards_history):
            for explorer in rewards:
                decayed_rewards[explorer] = (
                    decayed_rewards.get(explorer, 0.0) +
                    (rewards[explorer] * (gamma ** (len(rewards_history) - i))))
        return decayed_rewards


    def _new_sampling_weights(self, decayed_rewards):
        sampling_weights = []
        min_reward = min(decayed_rewards.values())
        max_reward = max(decayed_rewards.values())
        for e in self.explorers:
            sampling_weights.append((decayed_rewards[e.name] - min_reward) / (max_reward - min_reward))
            
        return np.exp(list(sampling_weights))/sum(np.exp(list(sampling_weights)))
        
        
    def optimize(self, softmax_temperature: float):
        """
        This is the function that you need to implement (including adding the necessary parameters)

        The P3BO population will consist of the three algorithms: adalead, random and genetic (they are in the
        optimizers directory). Each of these algorithms exposes a propose_sequences() method and a fit() method.

        You don't need to implement the adaptive variant of P3BO.
        """
        assert softmax_temperature > 0.0

        # Start with a uniform sampling across all the explorer algorithms.
        sampling_weights = np.ones(len(self.explorers)) / float(len(self.explorers))

        sequences: List[str] = []  # X in the paper.
        fitness_values: List[float] = []  # Y in the paper.

        lab_sequences_data: pd.DataFrame = self._initial_sequence_data(self.explorers[0])

        rewards_history: List[Dict[str, float]] = []  # to compute weight decay.
        for round in range(10):
            sequences_to_test: List[str] = []
            model_predictions: List[float] = []
            sequence_to_explorer = {}
            explorer_sequences = {}
            while len(sequences_to_test) < 10:
                explorer = np.random.choice(self.explorers, size=1, p=sampling_weights)[0]
                new_sequences, predictions = explorer.propose_sequences(
                    lab_sequences_data
                )
                # TODO(noelutz): Check that the sequences are sorted by predictions.
                # TODO(noelutz): Debug why the generic algorithm isn't returning any sequences.
                assert len(new_sequences) <= 10
                if len(new_sequences) == 0:
                    continue  # TODO(noelutz): Check what is happening here.

                # Add a single sequence to the set.
                for s in new_sequences:
                    if s not in sequences_to_test:
                        sequences_to_test.append(s)
                        model_predictions.append(s)
                        sequence_to_explorer[s] = explorer
                    explorer_sequences.setdefault(explorer.name, set()).add(s)
                    break
        
            # Run to the lab and get the real fitness value.
            true_scores = self.landscape.get_fitness(sequences_to_test)

            # Add the lab measurement to the list of sequences we track and pass in
            # when we generate new sequences.
            lab_sequences_data = lab_sequences_data.append(
                pd.DataFrame(
                    {
                        "sequence": sequences_to_test,
                        "model_score": model_predictions,
                        "true_score": true_scores,
                        "round": round,
                    }
                )
            )
            fmax = lab_sequences_data["true_score"].max()
            rewards: Dict[str, float] = self._get_rewards(fmax, sequences_to_test, true_scores, explorer_sequences)
            rewards_history.append(rewards)
            decayed_rewards = self._decayed_rewards(rewards_history)
            
            sampling_weights = self._new_sampling_weights(decayed_rewards)
                    
            # Re-train the different algorithms on the newly collected data.
            sequences.extend(sequences_to_test)
            fitness_values.extend(true_scores)
            for explorer in self.explorers:
                explorer.fit(sequences, fitness_values)
        return lab_sequences_data 


def main():
    # create a naive/mock model that simply computes the distance from the target optimum
    landscape = LevenstheinLandscape(optimal_sequence)
    model = NoisyAbstractModel(landscape)

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

    p3bo = P3bo([random, ga, adalead], landscape, sequences_batch_size=10)

    # that's the method you have to implement
    p3bo.optimize(0.9)


if __name__ == "__main__":
    main()
