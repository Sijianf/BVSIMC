# BVSIMC

Binary Variational Spike-and-slab Inductive Matrix Completion.

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
