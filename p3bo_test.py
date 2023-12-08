import pytest

import p3bo
import editdistance

def test_get_starting_sequence():
    assert p3bo.get_starting_sequence('MKYTKVM', 1.0) == 'MKYTKVM'
    assert editdistance.eval(p3bo.get_starting_sequence('KVMMKYTKVM', 0.8), 'KVMMKYTKVM') == 2
    assert editdistance.eval(p3bo.get_starting_sequence('KVM', 0.0), 'KVM') == 3
    assert editdistance.eval(p3bo.get_starting_sequence('K', 0.1), 'K') == 1
    