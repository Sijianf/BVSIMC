# BVSIMC

Python implementation of **BVSIMC** (**B**ayesian **V**ariable **S**election-Guided **I**nductive **M**atrix **C**ompletion), a matrix completion algorithm for binary target matrices that uses side-channel features (`U`, `V`) to infer missing entries.

BVSIMC places a spike-and-slab group lasso prior over the latent coefficients, giving it built-in variable selection: uninformative side-channel features are shrunk to exactly zero during fitting. This improves both predictive accuracy and interpretability — see the paper for validation on simulated data and two drug discovery applications (drug resistance prediction in *M. tuberculosis*, and drug-disease association prediction).

## Repository structure

- **bvsimc/** — the installable package. Contains the model implementation, its own README, and package-level usage notes.
- **examples/** — runnable demo notebooks: generate a synthetic dataset, fit BVSIMC, evaluate against held-out truth, and inspect which features were selected.

## Installation

```bash
cd bvsimc
pip install -e .
```

This installs `bvsimc` in editable mode, along with its dependencies (`numpy`, `scipy`, `scikit-learn`).

## Usage

```python
from bvsimc import BVSIMC

model = BVSIMC(
    Y, U, V,
    K=25,           # latent rank
    xi=10,          # positive-class weight
    eta=1e-4,       # step size
    lambda0=50, lambda1=5,          # spike / slab scale, V side
    tilde_lambda0=50, tilde_lambda1=5,  # spike / slab scale, U side
)

_, A, B, logLik = model.optimization(seed=0)
```

`A` and `B` are the fitted latent coefficient matrices — rows shrunk to all-zero were dropped by the spike-and-slab prior. Predicted probabilities are recovered via:

```python
from scipy.special import expit

prob = expit((U @ A) @ (V @ B).T)
```

## Examples

See `examples/demo_data_generation.ipynb` and `examples/demo_bvsimc.ipynb` for a full walkthrough: simulating data, fitting the model, evaluating AUC/AUPR, and checking feature-selection recovery against known ground truth.

## Citation

```bibtex
@article{fan2026bvsimc,
  title   = {BVSIMC: Bayesian Variable Selection-Guided Inductive Matrix Completion for Improved and Interpretable Drug Discovery},
  author  = {Fan, Sijian and Xiong, Liyan and Wang, Dayuan and Cai, Guoshuai and Bai, Ray},
  journal = {arXiv preprint arXiv:2603.18957},
  year    = {2026}
}
```

Paper: [arxiv.org/abs/2603.18957](https://arxiv.org/abs/2603.18957)

## License

MIT — see [LICENSE](LICENSE).
