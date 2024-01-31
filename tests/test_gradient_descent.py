import pytest
import numpy as np
from algorithms import gradient_descent
import os
print (os.getcwd())
@pytest.fixture
def optimizer():
    return gradient_descent.GradientDescent(learning_rate=0.01)

def test_update(optimizer):
    weights = np.array([1.0, 2.0, 3.0])
    gradients = np.array([0.5, 1.0, 1.5])
    expected_updated_weights = np.array([0.995, 1.99, 2.985])

    updated_weights = optimizer.update(weights, gradients)


    assert np.allclose(updated_weights, expected_updated_weights)