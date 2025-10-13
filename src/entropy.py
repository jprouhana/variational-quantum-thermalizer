"""
Entropy and free energy utilities for thermal state analysis.
"""

import numpy as np
from scipy.linalg import logm


def von_neumann_entropy(rho):
    """
    Compute the von Neumann entropy S(rho) = -Tr(rho * log(rho)).

    Uses eigenvalue decomposition to avoid log of singular matrices.

    Args:
        rho: density matrix (numpy array)

    Returns:
        entropy as a float
    """
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    # filter out zero/negative eigenvalues for numerical stability
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return -np.sum(eigenvalues * np.log(eigenvalues))


def relative_entropy(rho, sigma):
    """
    Compute the quantum relative entropy S(rho || sigma) = Tr(rho * (log(rho) - log(sigma))).

    Args:
        rho: density matrix (numpy array)
        sigma: density matrix (numpy array)

    Returns:
        relative entropy as a float
    """
    # regularize sigma to avoid log(0)
    eps = 1e-12
    sigma_reg = sigma + eps * np.eye(sigma.shape[0])

    log_rho = logm(rho + eps * np.eye(rho.shape[0]))
    log_sigma = logm(sigma_reg)

    result = np.real(np.trace(rho @ (log_rho - log_sigma)))
    return max(result, 0.0)  # relative entropy is non-negative


def free_energy(rho, H, beta):
    """
    Compute the variational free energy F = Tr(H * rho) - (1/beta) * S(rho).

    At the true Gibbs state, this equals the exact free energy -log(Z)/beta.

    Args:
        rho: density matrix (numpy array)
        H: Hamiltonian matrix (numpy array)
        beta: inverse temperature

    Returns:
        free energy as a float
    """
    energy = np.real(np.trace(H @ rho))
    entropy = von_neumann_entropy(rho)

    if beta < 1e-12:
        # infinite temperature limit — just minimize negative entropy
        return -entropy

    return energy - (1.0 / beta) * entropy
