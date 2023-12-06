"""Unit tests for p3bo."""

import random
import unittest

import editdistance

import p3bo


class MutationTest(unittest.TestCase):
    def setUp(self):
        # Fix random seed for deterministic / repeatable tests.
        random.seed(0)

    def test_mutate_with_alphabet(self):
        self.assertEqual(p3bo.mutate_with_alphabet(alphabet="AB", original="A"), "B")

    def test_get_starting_sequence(self):
        alphabet = "ABCDE"
        base_sequence = "AAAAAAAAAA"

        self.assertEqual(
            p3bo.get_starting_sequence(
                alphabet=alphabet, base_sequence=base_sequence, identity_percent=50
            ),
            "DAEADAEAAE",
        )

        self.assertEqual(
            editdistance.distance(
                base_sequence,
                p3bo.get_starting_sequence(
                    alphabet=alphabet, base_sequence=base_sequence, identity_percent=50
                ),
            ),
            5,
        )

        self.assertEqual(
            editdistance.distance(
                base_sequence,
                p3bo.get_starting_sequence(
                    alphabet=alphabet, base_sequence=base_sequence, identity_percent=20
                ),
            ),
            8,
        )

    def test_get_starting_sequence_preserves_identity(self):
        alphabet = "ABCDE"
        base_sequence = "AAAAAAAAAA"

        self.assertEqual(
            p3bo.get_starting_sequence(
                alphabet=alphabet, base_sequence=base_sequence, identity_percent=100
            ),
            base_sequence,
        )
