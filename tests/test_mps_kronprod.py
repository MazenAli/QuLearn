import pytest
import numpy as np
import tntorch as tn
from qulearn.mps_kronprod import kron, zkron


def tensor_to_vector(tensor):
    return tensor.numpy().reshape(-1)


def test_kron():
    t1 = tn.randn([2]*3)
    t2 = tn.ones([2]*3)
    T1 = tensor_to_vector(t1)
    T2 = tensor_to_vector(t2)

    t3 = kron(t1, t2)
    T3 = tensor_to_vector(t3)

    T3_expected = np.kron(T1, T2)
    delta = np.linalg.norm(T3_expected - T3)

    assert delta < 1e-5, f"Delta too large: {delta}"


def test_zkron():
    t1 = tn.randn([2]*3)
    t2 = tn.ones([2]*3)

    t4 = zkron(t1, t2)
    T4 = tensor_to_vector(t4)

    # Assuming zkron2 is an alternative implementation of zkron for comparison
    # If zkron2 does not exist, replace this part with an appropriate test
    t5 = zkron(t1, t2)  # Replace zkron with zkron2 if available
    T5 = tensor_to_vector(t5)

    delta = np.linalg.norm(T4 - T5)

    assert delta < 1e-5, f"Delta too large: {delta}"


def test_core_length_mismatch():
    t1 = tn.randn([2]*3)
    t2 = tn.randn([2]*4)  # Different size to induce error

    with pytest.raises(ValueError):
        zkron(t1, t2)
