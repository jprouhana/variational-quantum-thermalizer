"""
Hamiltonian construction and exact Gibbs state preparation.

Supports Ising and Heisenberg models for small qubit systems,
plus exact thermal state computation via matrix exponentiation.
"""

import numpy as np
from scipy.linalg import expm


# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_chain(ops):
    """Tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def ising_hamiltonian(n_qubits, J=1.0, h=0.5):
    """
    Build the transverse-field Ising Hamiltonian as a numpy matrix.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    Args:
        n_qubits: number of qubits (spins)
        J: coupling strength
        h: transverse field strength

    Returns:
        H as a 2^n x 2^n numpy array
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ interactions (nearest neighbor, open boundary)
    for i in range(n_qubits - 1):
        ops = [_I] * n_qubits
        ops[i] = _Z
        ops[i + 1] = _Z
        H -= J * _kron_chain(ops)

    # transverse field (X terms)
    for i in range(n_qubits):
        ops = [_I] * n_qubits
        ops[i] = _X
        H -= h * _kron_chain(ops)

    return H


def heisenberg_hamiltonian(n_qubits, J=1.0):
    """
    Build the Heisenberg XXX Hamiltonian as a numpy matrix.

    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    Args:
        n_qubits: number of qubits (spins)
        J: coupling strength

    Returns:
        H as a 2^n x 2^n numpy array
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(n_qubits - 1):
        for pauli in [_X, _Y, _Z]:
            ops = [_I] * n_qubits
            ops[i] = pauli
            ops[i + 1] = pauli
            H += J * _kron_chain(ops)

    return H


def exact_gibbs_state(H, beta):
    """
    Compute the exact Gibbs (thermal) state rho(beta) = exp(-beta*H) / Z.

    Args:
        H: Hamiltonian matrix (numpy array)
        beta: inverse temperature (1/kT)

    Returns:
        rho as a density matrix (numpy array)
    """
    rho_unnorm = expm(-beta * H)
    Z = np.real(np.trace(rho_unnorm))
    rho = rho_unnorm / Z
    return rho
