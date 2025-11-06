"""
Visualization functions for variational quantum thermalizer experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .entropy import von_neumann_entropy, free_energy


def plot_energy_vs_beta(results, save_dir='results'):
    """
    Plot expected energy and entropy vs inverse temperature beta.

    Shows both the variational and exact values side by side.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    betas = [r['beta'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- energy plot ---
    ax = axes[0]
    # need Hamiltonian to compute energy — we can get it from free energy + entropy
    # instead, store energies from density matrices
    exact_energies = []
    var_energies = []
    for r in results:
        # extract H from free energy computation
        rho_var = r['rho']
        rho_exact = r['rho_exact']
        # we need H to compute Tr(H*rho), but we can back it out from
        # F = E - S/beta => E = F + S/beta
        S_var = von_neumann_entropy(rho_var)
        S_exact = von_neumann_entropy(rho_exact)
        beta = r['beta']
        F_var = r['final_free_energy']
        E_var = F_var + S_var / beta if beta > 1e-12 else 0.0
        var_energies.append(E_var)

        # for exact, compute from last cost history or rho_exact
        # approximate from free_energy formula
        F_exact_approx = F_var - (F_var - r['cost_history'][-1]) if len(r['cost_history']) > 0 else F_var
        E_exact = F_exact_approx + S_exact / beta if beta > 1e-12 else 0.0
        exact_energies.append(E_exact)

    ax.plot(betas, var_energies, 'o-', color='#FF6B6B', linewidth=2,
            markersize=8, label='Variational')
    ax.plot(betas, exact_energies, 's--', color='#4ECDC4', linewidth=2,
            markersize=8, label='Exact')
    ax.set_xlabel('Inverse Temperature (beta)', fontsize=12)
    ax.set_ylabel('Energy <H>', fontsize=12)
    ax.set_title('Energy vs Inverse Temperature')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- entropy plot ---
    ax = axes[1]
    var_entropies = [von_neumann_entropy(r['rho']) for r in results]
    exact_entropies = [von_neumann_entropy(r['rho_exact']) for r in results]

    ax.plot(betas, var_entropies, 'o-', color='#FF6B6B', linewidth=2,
            markersize=8, label='Variational')
    ax.plot(betas, exact_entropies, 's--', color='#4ECDC4', linewidth=2,
            markersize=8, label='Exact')
    ax.set_xlabel('Inverse Temperature (beta)', fontsize=12)
    ax.set_ylabel('Von Neumann Entropy', fontsize=12)
    ax.set_title('Entropy vs Inverse Temperature')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- fidelity plot ---
    ax = axes[2]
    fidelities = [r['final_fidelity'] for r in results]

    ax.plot(betas, fidelities, 'o-', color='#45B7D1', linewidth=2,
            markersize=8)
    ax.set_xlabel('Inverse Temperature (beta)', fontsize=12)
    ax.set_ylabel('State Fidelity', fontsize=12)
    ax.set_title('Fidelity with Exact Gibbs State')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / 'energy_entropy_vs_beta.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'energy_entropy_vs_beta.png'}")


def plot_fidelity_convergence(history, title='Fidelity Convergence',
                                save_dir='results'):
    """Plot fidelity with exact Gibbs state over optimization iterations."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(history)), history, linewidth=1.5, color='#2196F3')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('State Fidelity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'fidelity_convergence.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'fidelity_convergence.png'}")


def plot_thermal_state_comparison(rho_exact, rho_var, save_dir='results'):
    """
    Side-by-side visualization of the exact and variational Gibbs states.
    Shows the real part of the density matrix as heatmaps.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # exact state
    im0 = axes[0].imshow(np.real(rho_exact), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0].set_title('Exact Gibbs State (Re)', fontsize=12)
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # variational state
    im1 = axes[1].imshow(np.real(rho_var), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1].set_title('Variational State (Re)', fontsize=12)
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # difference
    diff = np.real(rho_exact - rho_var)
    max_diff = max(np.abs(diff).max(), 1e-6)
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title('Difference (Exact - Variational)', fontsize=12)
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path / 'thermal_state_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'thermal_state_comparison.png'}")


def plot_free_energy_convergence(cost_history, exact_free_energy=None,
                                   save_dir='results'):
    """Plot the free energy cost function over optimization iterations."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(cost_history)), cost_history, linewidth=1.5,
            color='#FF6B6B', label='Variational')

    if exact_free_energy is not None:
        ax.axhline(y=exact_free_energy, color='#4ECDC4', linestyle='--',
                   linewidth=1.5, label=f'Exact ({exact_free_energy:.4f})')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Free Energy', fontsize=12)
    ax.set_title('Free Energy Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'free_energy_convergence.png', dpi=150)
    plt.close()
    print(f"Saved: {save_path / 'free_energy_convergence.png'}")
