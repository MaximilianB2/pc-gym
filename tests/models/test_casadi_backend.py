"""Regression tests ensuring every model is backend-agnostic under CasADi.

The MPC oracle (``oracle.setup_mpc``) evaluates each model with CasADi
symbolic variables and then ``vertcat``\\ s the returned derivatives. CasADi
``SX`` objects are *not* iterable, so any model that destructures its state or
input vector via Python tuple unpacking (``a, b, c = x``) raises

    Exception: CasADi matrices are not iterable by design.

These tests call every model with CasADi symbols of the correct shape -
mirroring the oracle's evaluation path - to guard against that regression.
"""

import inspect

import pytest
from casadi import SX, vertcat

import pcgym.model_classes as mc


def _model_classes():
    """All concrete model classes exposed by ``pcgym.model_classes``."""
    return {
        name: obj
        for name, obj in vars(mc).items()
        if inspect.isclass(obj) and obj.__module__ == mc.__name__ and name != "BaseModel" and hasattr(obj, "info")
    }


# ``coupled_oscillators`` relies on ``numpy.concatenate`` over symbolic
# entries, which is a separate (non-tuple-unpacking) CasADi limitation.
KNOWN_NON_CASADI = {"coupled_oscillators"}


@pytest.mark.parametrize("model_name", sorted(_model_classes()))
def test_model_callable_with_casadi_symbols(model_name):
    if model_name in KNOWN_NON_CASADI:
        pytest.xfail(f"{model_name} uses numpy.concatenate over symbolic types (separate CasADi limitation)")

    model = _model_classes()[model_name](int_method="casadi")
    info = model.info()
    n_x = len(info["states"])
    n_u = len(info["inputs"])

    x = SX.sym("x", n_x)
    if n_u > 0:
        u = SX.sym("u", n_u)
        derivatives = model(x, u)
    else:
        derivatives = model(x)

    # The oracle vertcats the returned derivatives; this is where a
    # tuple-unpacked model would already have raised.
    dx = vertcat(*derivatives)
    assert dx.shape[0] == n_x, f"{model_name} returned {dx.shape[0]} derivatives for {n_x} states"


if __name__ == "__main__":
    pytest.main([__file__])
