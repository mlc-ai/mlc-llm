# MLC LLM Tests

We primarily relies on pytest to test our engine.
Most of the unit functionalities in C++ can be exposed via TVM FFI,
and tested through python environment.

We categorize the test cases by adding `pytestmark = [pytest.mark.category_name]`.
Checkout [python/conftest.py](python/conftest.py) for categories.
