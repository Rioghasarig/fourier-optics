#!/usr/bin/env python3
"""
Rayleigh-Sommerfeld Diffraction Simulation
==========================================
Coordinate system:
  - x : toward the reader
  - y : to the right  (optical axis)
  - z : up

Geometry:
  - Planar screen in the x–z plane (y = 0) with a circular aperture of
    radius r_ap centered at the origin
  - Point source P2 at y < 0  (to the left of the screen)
  - Observation points P0 at y > 0  (to the right of the screen)

RS first-solution formula (Dirichlet Green's function):

    U(P0) = (1/iλ) ∬_Σ U_inc(P1) · [e^(ikr01)/r01] · cos θ  dA

where:
    U_inc(P1) = A · e^(ikr21) / r21        spherical wave at aperture point P1
    r21        = |P1 − P2|
    r01        = |P0 − P1|
    cos θ      = y0 / r01                  obliquity factor (screen normal = ŷ)
    λ          = 2π/k

The integral is discretised with a uniform grid over the aperture square
[-r_ap, r_ap]² and the circular mask is applied analytically.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─── Parameters ────────────────────────────────────────────────────────────────
lam  = 500e-9                            # wavelength  [m]
k    = 2 * np.pi / lam                   # wavenumber  [rad/m]
A    = 1.0                               # source amplitude  (arbitrary units)
r_ap = 1e-3                              # aperture radius  [m]
P2   = np.array([0.0, -50e-2, 0.0])     # source: on-axis, 50 mm behind screen

# Primary observation plane
obs_y    = 20e-3                         # distance beyond screen  [m]
obs_half = 3 * r_ap                      # ±half-width of observation window  [m]
N_obs    = 400                            # obs-grid size per side

# Aperture quadrature: increase N_ap for accuracy, decrease for speed.
# Rule of thumb: N_ap ≥ 2 · r_ap · sin(θ_max) / λ  where θ_max = r_ap / obs_y
N_ap = 400



# ─── Core integral ──────────────────────────────────────────────────────────────
_worker_state: dict = {}


def _init_worker(y0, Z0, X1_ap, Z1_ap, Uap_ap, k, ds):
    """Pool initializer: store shared arrays once per worker process."""
    _worker_state['y0']     = y0
    _worker_state['Z0']     = Z0
    _worker_state['X1_ap']  = X1_ap
    _worker_state['Z1_ap']  = Z1_ap
    _worker_state['Uap_ap'] = Uap_ap
    _worker_state['k']      = k
    _worker_state['ds']     = ds


def _rs_column(x0):
    """Worker: RS integral for a single x observation coordinate."""
    y0, Z0     = _worker_state['y0'], _worker_state['Z0']
    X1_ap      = _worker_state['X1_ap']
    Z1_ap      = _worker_state['Z1_ap']
    Uap_ap     = _worker_state['Uap_ap']
    k, ds      = _worker_state['k'], _worker_state['ds']
    wavelength = 2 * np.pi / k
    r01        = np.sqrt((X1_ap - x0)**2 + y0**2 + (Z0 - Z1_ap)**2)
    cos_theta  = y0 / r01
    integrand  = Uap_ap * np.exp(1j * k * r01) / r01 * cos_theta / (1j * wavelength)
    return np.sum(integrand, axis=-1) * ds**2


def _uap_chunk(args):
    """Compute U_ap for a slice of rows [i_start, i_stop)."""
    i_start, i_stop, N_asm, ds_asm, x2, y2, z2, r_ap, A, k = args
    s_full = (np.arange(N_asm) - N_asm // 2) * ds_asm
    s_z    = s_full[i_start:i_stop]
    X_c, Z_c = np.meshgrid(s_full, s_z)
    r21 = np.sqrt((X_c - x2)**2 + y2**2 + (Z_c - z2)**2)
    return np.where(X_c**2 + Z_c**2 <= r_ap**2,
                    A * np.exp(1j * k * r21) / r21,
                    0+0j)


def compute_uap_parallel(N_asm, ds_asm, P2, r_ap, A, k, n_workers=None):
    """Build the aperture field in parallel by splitting rows across workers."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    x2, y2, z2 = P2
    chunk = (N_asm + n_workers - 1) // n_workers
    tasks = [
        (i, min(i + chunk, N_asm), N_asm, ds_asm, x2, y2, z2, r_ap, A, k)
        for i in range(0, N_asm, chunk)
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunks = list(tqdm(executor.map(_uap_chunk, tasks),
                           total=len(tasks), desc="U_ap"))
    return np.vstack(chunks)


def _aperture_grid(r_ap: float, N_ap: int):
    """Uniform grid covering the aperture square with a circular mask."""
    s  = np.linspace(-r_ap, r_ap, N_ap)
    ds = s[1] - s[0]
    X1, Z1 = np.meshgrid(s, s)
    mask = (X1**2 + Z1**2) <= r_ap**2
    return X1, Z1, mask, ds


def rs_single(P0, P2, k, A, r_ap, N_ap=200):
    """
    RS phasor at a single observation point P0.

    Parameters
    ----------
    P0 : (3,) array-like  — observation point, y > 0
    P2 : (3,) array-like  — source point, y < 0
    k  : float            — wavenumber
    A  : float            — source amplitude
    r_ap : float          — aperture radius
    N_ap : int            — quadrature grid points per side

    Returns
    -------
    complex scalar
    """
    x0, y0, z0 = P0
    x2, y2, z2 = P2
    wavelength = 2 * np.pi / k

    X1, Z1, mask, ds = _aperture_grid(r_ap, N_ap)

    r21 = np.sqrt((X1 - x2)**2 + y2**2 + (Z1 - z2)**2)
    r01 = np.sqrt((X1 - x0)**2 + y0**2 + (Z1 - z0)**2)

    U_inc     = A * np.exp(1j * k * r21) / r21
    cos_theta = y0 / r01                        # screen normal = ŷ  (points right)
    integrand = U_inc * np.exp(1j * k * r01) / r01 * cos_theta / (1j * wavelength)

    return np.sum(np.where(mask, integrand, 0.0)) * ds**2


def rs_plane(y0, x_obs, z_obs, k, U_ap, X1, Z1, ds, n_workers=None):
    """
    RS phasor field on a 2-D observation plane at fixed depth y0.

    Only sums over aperture points where U_ap is non-zero, so the grid
    may be larger than the aperture without a performance penalty.
    The x-loop is parallelised across n_workers processes.

    Parameters
    ----------
    y0         : float          — observation plane depth along y-axis
    x_obs      : (Nx,) array    — x coordinates on observation plane
    z_obs      : (Nz,) array    — z coordinates on observation plane
    k          : float          — wavenumber [rad/m]
    U_ap       : (N, N) complex — aperture field (zeros outside aperture)
    X1         : (N, N) float   — x grid coordinates matching U_ap
    Z1         : (N, N) float   — z grid coordinates matching U_ap
    ds         : float          — grid spacing [m]
    n_workers  : int or None    — parallel workers (default: os.cpu_count())

    Returns
    -------
    U : complex ndarray, shape (Nz, Nx)
    """
    # Flatten to aperture-only points to keep memory bounded
    ap     = U_ap != 0
    X1_ap  = X1[ap]    # (N_eff,)
    Z1_ap  = Z1[ap]    # (N_eff,)
    Uap_ap = U_ap[ap]  # (N_eff,)
    Z0     = z_obs[:, np.newaxis]  # (Nz, 1)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Each task is a single x column; shared arrays are sent once per worker via
    # the initializer so they are not re-pickled for every task.
    initargs = (y0, Z0, X1_ap, Z1_ap, Uap_ap, k, ds)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=initargs) as executor:
        cols = list(tqdm(executor.map(_rs_column, x_obs),
                         total=len(x_obs), desc="rs_plane"))

    return np.column_stack(cols)

def asm_plane(U_ap, ds, y0, k):
    """
    Propagate a 2-D field by distance y0 using the angular spectrum method.

    U_ap must be centered in the array (index [N//2, N//2] = spatial origin).
    ifftshift/fftshift move the origin to/from the FFT's expected [0,0] corner.

    Parameters
    ----------
    U_ap : (N, N) complex array — field at source plane, centered in array
    ds   : float               — grid spacing [m]
    y0   : float               — propagation distance [m]
    k    : float               — wavenumber [rad/m]

    Returns
    -------
    U_prop : (N, N) complex array — propagated field, same grid as U_ap
    """
    N  = U_ap.shape[0]
    fx = np.fft.fftfreq(N, d=ds)
    Fx, Fz = np.meshgrid(fx, fx)

  
    ky_sq = k**2 - ( 2 * np.pi * Fx )**2 - (2 * np.pi * Fz)**2
    ky    = np.sqrt(np.maximum(ky_sq, 0.0))

    H = np.exp(1j * ky * y0)
    H[ky_sq < 0] = 0.0   # zero evanescent components

    A_spec = np.fft.fft2(np.fft.ifftshift(U_ap))
    return np.fft.fftshift(np.fft.ifft2(A_spec * H))


# ─── Shared aperture field (used by both RS and ASM) ───────────────────────────
# Grid satisfies ds ≤ λ/(2 sin θ_max) ≈ 12.5 µm and spans > ±obs_half.
ds_asm = 2e-6    # 8 µm, safely below the 12.5 µm Nyquist limit
N_asm  = 4096    # power-of-2 grid; total extent ≈ 8.19 mm (> 2 × obs_half)

s_asm        = (np.arange(N_asm) - N_asm // 2) * ds_asm   # centered coordinates
X_asm, Z_asm = np.meshgrid(s_asm, s_asm)

U_ap = compute_uap_parallel(N_asm, ds_asm, P2, r_ap, A, k)
print(f"U_ap constructed: shape={U_ap.shape}, non-zero points={np.count_nonzero(U_ap)}")


if __name__ == '__main__':
    # ─── RS simulation ─────────────────────────────────────────────────────────
    x_obs = np.linspace(-obs_half, obs_half, N_obs)
    z_obs = np.linspace(-obs_half, obs_half, N_obs)
    U_rs  = rs_plane(obs_y, x_obs, z_obs, k, U_ap, X_asm, Z_asm, ds_asm)

    # ─── ASM simulation ────────────────────────────────────────────────────────
    U_asm_full = asm_plane(U_ap, ds_asm, obs_y, k)

    # Crop to the same ±obs_half window used by RS
    i_lo  = np.searchsorted(s_asm, -obs_half)
    i_hi  = np.searchsorted(s_asm,  obs_half)
    U_asm = U_asm_full[i_lo:i_hi, i_lo:i_hi]
    s_roi = s_asm[i_lo:i_hi]

    # ─── Comparison plot ───────────────────────────────────────────────────────
    mm      = 1e-3
    ext_rs  = np.array([-obs_half, obs_half, -obs_half, obs_half]) / mm
    ext_asm = np.array([s_roi[0], s_roi[-1], s_roi[0], s_roi[-1]]) / mm

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    (ax_rs_amp, ax_rs_ph), (ax_asm_amp, ax_asm_ph) = axes

    # RS row
    im = ax_rs_amp.imshow(np.abs(U_rs), extent=ext_rs, origin='lower',
                          cmap='inferno', aspect='equal')
    ax_rs_amp.set_title("|U|  — Rayleigh-Sommerfeld")
    ax_rs_amp.set_xlabel("x  [mm]")
    ax_rs_amp.set_ylabel("z  [mm]")
    fig.colorbar(im, ax=ax_rs_amp, fraction=0.046)

    im = ax_rs_ph.imshow(np.angle(U_rs), extent=ext_rs, origin='lower',
                         cmap='hsv', vmin=-np.pi, vmax=np.pi, aspect='equal')
    ax_rs_ph.set_title("∠U  — Rayleigh-Sommerfeld")
    ax_rs_ph.set_xlabel("x  [mm]")
    ax_rs_ph.set_ylabel("z  [mm]")
    fig.colorbar(im, ax=ax_rs_ph, label="[rad]", fraction=0.046)

    # ASM row
    im = ax_asm_amp.imshow(np.abs(U_asm), extent=ext_asm, origin='lower',
                           cmap='inferno', aspect='equal')
    ax_asm_amp.set_title("|U|  — Angular Spectrum")
    ax_asm_amp.set_xlabel("x  [mm]")
    ax_asm_amp.set_ylabel("z  [mm]")
    fig.colorbar(im, ax=ax_asm_amp, fraction=0.046)

    im = ax_asm_ph.imshow(np.angle(U_asm), extent=ext_asm, origin='lower',
                          cmap='hsv', vmin=-np.pi, vmax=np.pi, aspect='equal')
    ax_asm_ph.set_title("∠U  — Angular Spectrum")
    ax_asm_ph.set_xlabel("x  [mm]")
    ax_asm_ph.set_ylabel("z  [mm]")
    fig.colorbar(im, ax=ax_asm_ph, label="[rad]", fraction=0.046)

    fig.suptitle(
        f"Diffraction at y = {obs_y*1e3:.0f} mm  "
        f"(λ = {lam*1e9:.0f} nm,  r_ap = {r_ap*1e3:.0f} mm)",
        fontweight='bold')
    plt.tight_layout()

    out = "diffraction.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {out}")
