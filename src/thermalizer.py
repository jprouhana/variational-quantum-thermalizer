"""
Variational Quantum Thermalizer — prepares approximate Gibbs states
using a parameterized quantum circuit and free energy minimization.

References:
    Wu & Hsieh (2019) - "Variational Thermal Quantum Simulation"
    Verdon et al. (2019) - "Quantum Hamiltonian-Based Models & the
        Variational Quantum Thermalizer Algorithm"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity
from scipy.optimize import minimize

from .entropy import von_neumann_entropy, free_energy


class VariationalThermalizer:
    """
    Prepares approximate thermal (Gibbs) states rho(beta) = exp(-beta*H)/Z
    using a variational circuit that minimizes the free energy
    F = Tr(H*rho) + (1/beta)*S(rho).

    The ansatz uses layers of parameterized rotations and entangling gates.
    """

    def __init__(self, n_qubits, n_layers=3, shots=4096, seed=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.seed = seed
        self.simulator = AerSimulator(method='statevector')

        # total parameters: 3 rotations per qubit per layer
        self.n_params = n_qubits * 3 * n_layers

        # optimization tracking
        self.cost_history = []
        self.fidelity_history = []

    def _build_ansatz(self, params):
        """
        Build the parameterized ansatz circuit.

        Uses Ry-Rz-Ry rotations on each qubit followed by a ladder
        of CNOT gates for entanglement, repeated for each layer.
        """
        qc = QuantumCircuit(self.n_qubits)
        idx = 0

        for layer in range(self.n_layers):
            # single-qubit rotations
            for q in range(self.n_qubits):
                qc.ry(params[idx], q)
                idx += 1
                qc.rz(params[idx], q)
                idx += 1
                qc.ry(params[idx], q)
                idx += 1

            # entangling layer
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)

        return qc

    def _get_density_matrix(self, params):
        """
        Run the ansatz and extract the density matrix from statevector sim.
        """
        qc = self._build_ansatz(params)
        qc.save_statevector()

        job = self.simulator.run(qc, seed_simulator=self.seed)
        statevector = job.result().get_statevector()

        # convert to density matrix
        rho = np.outer(np.array(statevector), np.conj(np.array(statevector)))
        return rho

    def _cost_function(self, params, H, beta, rho_exact=None):
        """
        Variational free energy cost.

        F = Tr(H * rho) + (1/beta) * S(rho)
        Wait — free energy is E - T*S = Tr(H*rho) - (1/beta)*S(rho).
        We want to minimize this.
        """
        rho = self._get_density_matrix(params)
        cost = free_energy(rho, H, beta)
        self.cost_history.append(cost)

        # track fidelity with exact state if available
        if rho_exact is not None:
            fid = np.real(state_fidelity(DensityMatrix(rho),
                                          DensityMatrix(rho_exact)))
            self.fidelity_history.append(fid)

        return cost

    def train(self, H, beta, maxiter=200, seed=None):
        """
        Train the variational thermalizer to prepare the Gibbs state at
        inverse temperature beta.

        Args:
            H: Hamiltonian matrix (numpy array)
            beta: inverse temperature
            maxiter: maximum optimization iterations
            seed: random seed for initial parameters

        Returns:
            dict with 'rho', 'optimal_params', 'cost_history',
            'fidelity_history', 'final_free_energy', 'final_fidelity'
        """
        from .hamiltonians import exact_gibbs_state

        self.cost_history = []
        self.fidelity_history = []

        # compute exact Gibbs state for fidelity tracking
        rho_exact = exact_gibbs_state(H, beta)

        # random initial parameters
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        x0 = rng.uniform(0, 2 * np.pi, self.n_params)

        # optimize
        res = minimize(
            self._cost_function,
            x0,
            args=(H, beta, rho_exact),
            method='COBYLA',
            options={'maxiter': maxiter, 'rhobeg': 0.5}
        )

        # get the final density matrix
        rho_final = self._get_density_matrix(res.x)
        final_fidelity = np.real(state_fidelity(DensityMatrix(rho_final),
                                                  DensityMatrix(rho_exact)))

        return {
            'rho': rho_final,
            'rho_exact': rho_exact,
            'optimal_params': res.x,
            'cost_history': self.cost_history.copy(),
            'fidelity_history': self.fidelity_history.copy(),
            'final_free_energy': self.cost_history[-1],
            'final_fidelity': final_fidelity,
            'optimization_result': res,
        }


def run_temperature_sweep(H, betas, n_qubits, n_layers=3, maxiter=200,
                           seed=None):
    """
    Run the thermalizer at multiple inverse temperatures.

    Returns a list of result dicts, one per beta value.
    """
    results = []

    for beta in betas:
        print(f"Training at beta={beta:.2f}...")
        thermalizer = VariationalThermalizer(
            n_qubits, n_layers=n_layers, seed=seed
        )
        result = thermalizer.train(H, beta, maxiter=maxiter, seed=seed)
        result['beta'] = beta
        results.append(result)
        print(f"  fidelity={result['final_fidelity']:.4f}, "
              f"free_energy={result['final_free_energy']:.4f}")

    return results
