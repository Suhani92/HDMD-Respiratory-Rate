#!/usr/bin/env python3
"""
VMD baseline breathing estimation
Usage:
    python vmd_breathing_estimation.py --file path/to/h5
"""
import argparse
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.signal import welch
import time
import os


def load_and_preprocess(file_path, fs_desired=300, duration_desired=60):
    with h5py.File(file_path, "r") as f:
        frame = f["sessions/session_0/group_0/entry_0/result/frame"]
        real_part = np.array(frame["real"], dtype=np.float64)
        imag_part = np.array(frame["imag"], dtype=np.float64)

    IQ_data = (real_part + 1j * imag_part).transpose(2, 1, 0)
    num_sweeps = IQ_data.shape[2]
    original_duration = 60.0
    original_fs = num_sweeps / original_duration

    # Range bin selection
    mag = np.abs(IQ_data)
    mean_mag = np.mean(mag, axis=2)
    peak_idx = np.argmax(mean_mag, axis=1)
    r0 = max(0, peak_idx[0] - 5)
    r1 = min(IQ_data.shape[1], peak_idx[0] + 5)
    sel_bins = np.arange(r0, r1 + 1)

    # Low-pass filtering
    tau_iq = 0.04
    alpha_iq = np.exp(-2 / (tau_iq * original_fs))
    down = IQ_data[:, sel_bins, :]
    filtered = np.zeros_like(down)
    filtered[:, :, 0] = down[:, :, 0]
    for s in range(1, down.shape[2]):
        filtered[:, :, s] = alpha_iq * filtered[:, :, s - 1] + (1 - alpha_iq) * down[:, :, s]

    # Phase extraction
    f_low = 0.2
    alpha_phi = np.exp(-2 * f_low / original_fs)
    phi = np.zeros(filtered.shape[2])
    for s in range(1, filtered.shape[2]):
        z = np.sum(filtered[:, :, s] * np.conj(filtered[:, :, s - 1]))
        phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

    t_old = np.linspace(0, original_duration, len(phi), endpoint=False)
    interp_func = interp1d(t_old, phi, kind='cubic', fill_value="extrapolate")
    new_N = int(duration_desired * fs_desired)
    t_new = np.linspace(0, duration_desired, new_N, endpoint=False)
    phi_resampled = interp_func(t_new)

    return phi_resampled, fs_desired


def vmd_improved(signal, K=6, alpha=2000, tau=0, init=2, tol=1e-6, max_iter=200, fs=300.0):
    """
    Simple VMD implementation adapted for our use (mirror-extension + iterative update).
    Returns: modes (K x T), center frequencies (omega in Hz)
    """
    T = len(signal)
    f_mirror = np.concatenate([signal[::-1], signal, signal[::-1]])   
    N = len(f_mirror)
    freqs = np.fft.fftfreq(N, d=1.0/fs)
    freqs_shift = np.fft.fftshift(freqs)

    if init == 1:
        omega = np.linspace(0.05, 0.45, K)
    else:
        omega = np.zeros(K)
        base = np.linspace(0.05, 0.45, K)
        omega[:] = base

    u_hat = np.zeros((K, N), dtype=complex)
    lambda_hat = np.zeros(N, dtype=complex)
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))

    omega_old = omega.copy()
    n_iter = 0
    eps = 1e-9

    while n_iter < max_iter:
        for k in range(K):
            sum_other = np.sum(u_hat, axis=0) - u_hat[k, :]
            numerator = f_hat - sum_other - lambda_hat / 2.0
            denominator = 1.0 + alpha * (freqs_shift - omega[k]) ** 2
            u_hat[k, :] = numerator / (denominator + eps)
            power = np.abs(u_hat[k, :]) ** 2
            omega[k] = np.sum(freqs_shift * power) / (np.sum(power) + eps)
            omega[k] = np.clip(omega[k], 0.0, fs / 2.0)

        residual = f_hat - np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * residual

        if n_iter > 0:
            diff = np.sum(np.abs(omega - omega_old))
            if diff < tol:
                break
        omega_old = omega.copy()
        n_iter += 1

    # Reconstruct modes (time domain), extract middle segment
    modes = np.zeros((K, T))
    for k in range(K):
        temp = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))
        modes[k, :] = temp[T:2 * T]

    return modes, omega


def estimate_breathing_vmd(file_path, K=6, alpha=2000, fs_desired=300):
    phi, fs = load_and_preprocess(file_path, fs_desired=fs_desired)
    start = time.time()
    modes, omega = vmd_improved(phi, K=K, alpha=alpha, init=2, fs=fs)
    elapsed = time.time() - start

    breathing_modes = [i for i, f in enumerate(omega) if 0.1 <= f <= 0.8]
    if not breathing_modes:
        target = 0.3
        diffs = np.abs(omega - target)
        idx = int(np.argmin(diffs))
    else:
        energies = [np.sum(np.abs(modes[i, :]) ** 2) for i in breathing_modes]
        idx = breathing_modes[int(np.argmax(energies))]

    breathing_freq_hz = omega[idx]
    if not (0.1 <= breathing_freq_hz <= 0.8):
        breathing_freq_hz = 0.3  

    bpm = breathing_freq_hz * 60.0
    reconstructed = modes[idx, :]
    return bpm, reconstructed, elapsed


def main():
    parser = argparse.ArgumentParser(description="VMD Baseline breathing estimation")
    parser.add_argument("--file", required=True, help="Path to .h5 radar file")
    parser.add_argument("--K", type=int, default=6, help="Number of VMD modes")
    parser.add_argument("--alpha", type=float, default=2000.0, help="VMD alpha")
    parser.add_argument("--fs", type=float, default=300.0, help="Desired resampling rate")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    print(f"Processing {args.file} with VMD (K={args.K}, alpha={args.alpha})...")
    bpm, recon, elapsed = estimate_breathing_vmd(args.file, K=args.K, alpha=args.alpha, fs_desired=args.fs)
    print(f"VMD Breathing Rate: {bpm:.2f} BPM | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
