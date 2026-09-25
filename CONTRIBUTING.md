# Contributing

regimelib is developed at [github.com/microprediction/regimelib](https://github.com/microprediction/regimelib).
Bug reports, questions and feature requests go through the issue tracker; changes through pull requests.

## The rule a change must satisfy

Every model has a *frozen limit* (all regimes equal) in which it is QuantLib's model, and every engine has a
*referee* with switching on: the numerical solution of the reduced system, the coupled PDE on a grid, or Monte
Carlo with exact regime paths. A pull request that adds a model, an instrument or an engine adds a test in
`tests/` with both checks. A change that alters a number changes the test that certifies it, and says why.

## Where things go

- `regimelib/models.py`: a model is a statement of its forcing (`bondForcing`, `returnForcing`) and, for the grid
  engines, its operators. Exact reductions only; a model whose switched operators do not commute with the rest
  goes in the first-order tier (`FirstOrderFDEngine`).
- `regimelib/instruments.py`: QuantLib's names and parameters, with a chain in front.
- `regimelib/engines.py`, `fd.py`, `montecarlo.py`, `firstorder.py`: engines.
- `regimelib/symbolic.py`: closed forms as sympy expressions, each checked against the engine at the same order.
- `docs/`: Sphinx source, in QuantLib-Python's layout; every public class gets a signature, a description and an
  example.

## Running the tests

    pip install -e ".[test]"
    pytest -m "not slow"     # about two minutes
    pytest                   # everything, about a quarter of an hour

## Style

Match the surrounding code. Comments say what a formula is and where it comes from, not what the code does.
