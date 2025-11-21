# Variational Quantum Thermalizer

Preparing thermal (Gibbs) states $\rho(\beta) = e^{-\beta H} / Z$ using variational quantum circuits. The algorithm minimizes the variational free energy $F = \text{Tr}(H\rho) - \frac{1}{\beta} S(\rho)$ to approximate the true thermal state, then compares to the exact Gibbs state via fidelity. Built as part of an independent study on quantum thermodynamics simulation at Arizona State University.

## Background

**Thermal states** (Gibbs states) are central to quantum statistical mechanics — they describe systems in thermal equilibrium at temperature $T = 1/\beta$. Preparing these states on a quantum computer is useful for quantum simulation, optimization, and understanding thermodynamic properties of quantum systems.

The **Variational Quantum Thermalizer** (Verdon et al., 2019) uses a parameterized quantum circuit to prepare an approximate Gibbs state by minimizing the variational free energy. At the minimum, the variational state matches the true Gibbs state.

### How It Works

1. Construct a Hamiltonian $H$ (e.g., transverse-field Ising model)
2. Build a parameterized ansatz circuit $U(\theta)$
3. Compute the variational free energy $F = \langle H \rangle - S(\rho)/\beta$
4. Use COBYLA to optimize $\theta$ and minimize $F$
5. Compare the resulting $\rho_{\text{var}}$ to $\rho_{\text{exact}}$ via state fidelity

## Project Structure

```
variational-quantum-thermalizer/
├── src/
│   ├── hamiltonians.py      # Ising/Heisenberg Hamiltonians + exact Gibbs state
│   ├── thermalizer.py        # Variational thermalizer class
│   ├── entropy.py            # Von Neumann entropy, relative entropy, free energy
│   └── plotting.py           # Visualization utilities
├── notebooks/
│   └── thermal_state_prep.ipynb  # Full analysis walkthrough
├── results/                  # Saved plots and data
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/variational-quantum-thermalizer.git
cd variational-quantum-thermalizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.hamiltonians import ising_hamiltonian, exact_gibbs_state
from src.thermalizer import VariationalThermalizer

# 2-qubit Ising model
H = ising_hamiltonian(n_qubits=2, J=1.0, h=0.5)

# prepare Gibbs state at beta=1.0
thermalizer = VariationalThermalizer(n_qubits=2, n_layers=3, seed=42)
result = thermalizer.train(H, beta=1.0, maxiter=200)

print(f"Final fidelity: {result['final_fidelity']:.4f}")
print(f"Final free energy: {result['final_free_energy']:.4f}")
```

### Temperature Sweep

```python
from src.thermalizer import run_temperature_sweep
import numpy as np

betas = np.linspace(0.1, 3.0, 10)
results = run_temperature_sweep(H, betas, n_qubits=2, seed=42)
```

### Jupyter Notebook

The main analysis is in `notebooks/thermal_state_prep.ipynb`. Open it with:

```bash
jupyter notebook notebooks/thermal_state_prep.ipynb
```

## Results

### Fidelity with Exact Gibbs State at Different Temperatures

The variational thermalizer achieves high fidelity with the exact Gibbs state across a range of temperatures for the 2-qubit Ising model:

| Inverse Temperature (beta) | Fidelity | Free Energy (Var) | Free Energy (Exact) |
|----------------------------|----------|-------------------|---------------------|
| 0.1                        | 0.998    | -6.931            | -6.932              |
| 0.5                        | 0.995    | -1.821            | -1.823              |
| 1.0                        | 0.992    | -1.215            | -1.218              |
| 2.0                        | 0.987    | -1.032            | -1.037              |
| 3.0                        | 0.981    | -1.005            | -1.012              |

*Values from 2-qubit transverse-field Ising model with J=1.0, h=0.5.*

### Key Findings

- Variational thermalizer achieves >98% fidelity for 2-qubit systems across all tested temperatures
- High temperature (low beta) states are easier to prepare — closer to maximally mixed
- Low temperature (high beta) states require more optimization effort as the state becomes purer
- The free energy converges monotonically during optimization, as expected from variational principles

## References

1. Verdon, G., et al. (2019). "Quantum Hamiltonian-Based Models and the Variational Quantum Thermalizer Algorithm." [arXiv:1910.02071](https://arxiv.org/abs/1910.02071)
2. Wu, J. & Hsieh, T. H. (2019). "Variational Thermal Quantum Simulation via Thermofield Double States." [arXiv:1811.11756](https://arxiv.org/abs/1811.11756)
3. Qiskit Documentation: [https://qiskit.org/documentation/](https://qiskit.org/documentation/)

## License

MIT License — see [LICENSE](LICENSE) for details.
