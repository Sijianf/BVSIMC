# BVSIMC

Bayesian Variable Selection-Guided Inductive Matrix Completion.

Paper: [arxiv.org/abs/2603.18957](https://arxiv.org/abs/2603.18957)

## Install

```bash
pip install -e .
```

## Quick start

```python
from bvsimc import BVSIMC

model = BVSIMC(Y, U, V, K=10)
mu, A, B, logLik = model.optimization(seed=42)
```

## Examples

See `examples/` for simulation and real-data walkthroughs.
