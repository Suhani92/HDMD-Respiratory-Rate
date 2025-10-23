#!/usr/bin/env python3
"""
Hankel Dynamic Mode Decomposition (HDMD) for Breathing Rate Estimation
This script performs breathing rate estimation using Hankel-DMD on radar IQ data. 

Usage:
------
# Run HDMD on a folder of radar .h5 files
python3 dmdt_tracking.py --folder /path/to/radar_data --output results.csv

# Optionally save the extracted phase waveforms
python3 dmdt_tracking.py --folder /path/to/radar_data --save_waveform

# Run on synthetic signals
python3 dmdt_tracking.py --folder ./synthetic_data --mode synthetic
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.linalg import hankel, svd, eig
from scipy.interpolate import interp1d
import h5py
import time
import os
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 1


def hankel_dmd(X, Y, fs, rank=None, tol=1e-10):
    U, s, Vh = svd(X, full_matrices=False)
    if rank is None:
        rank = np.sum(s > tol * s[0])

    U_r, s_r, Vh_r = U[:, :rank], s[:rank], Vh[:rank, :]
    A_tilde = U_r.conj().T @ Y @ Vh_r.conj().T @ np.diag(1 / s_r)
    evals, evecs = eig(A_tilde)
    modes = Y @ Vh_r.conj().T @ np.diag(1 / s_r) @ evecs

    dt = 1 / fs
    omega = np.log(evals) / dt
    frequencies = np.imag(omega) / (2 * np.pi)
    damping_rates = np.real(omega)
    return modes, evals, frequencies, damping_rates


def diagonal_averaging(H_matrix):
    m, n = H_matrix.shape
    L = m + n - 1
    result = np.zeros(L)
    for k in range(L):
        elements = []
        for i in range(max(0, k - n + 1), min(m, k + 1)):
            j = k - i
            if 0 <= j < n:
                elements.append(H_matrix[i, j])
        if elements:
            result[k] = np.mean(elements)
    return result


def reconstruct_from_modes(selected_modes, X, modes, evals, fs):
    modes_sel = modes[:, selected_modes]
    b = np.linalg.pinv(modes_sel) @ X[:, 0]
    dt = 1 / fs
    omega_sel = np.log(evals[selected_modes]) / dt
    time_steps = np.arange(X.shape[1])
    time_dynamics = np.zeros((len(selected_modes), len(time_steps)), dtype=complex)
    for i, idx in enumerate(selected_modes):
        time_dynamics[i, :] = b[i] * np.exp(omega_sel[i] * time_steps * dt)
    X_dmd = np.real(modes_sel @ time_dynamics)
    return diagonal_averaging(X_dmd)


def process_file(file_path, fs_desired=300, duration_desired=60):
    with h5py.File(file_path, "r") as f:
        frame = f["sessions/session_0/group_0/entry_0/result/frame"]
        real_part = np.array(frame["real"], dtype=np.float64)
        imag_part = np.array(frame["imag"], dtype=np.float64)

    IQ_data = (real_part + 1j * imag_part).transpose(2, 1, 0)  # (range, bins, sweeps)
    num_sweeps = IQ_data.shape[2]
    original_duration = 60
    original_fs = num_sweeps / original_duration

    # Range bin selection
    magnitude = np.abs(IQ_data)
    mean_magnitude = np.mean(magnitude, axis=(0, 2))
    peak_idx = np.argmax(mean_magnitude)
    range_start = max(0, peak_idx - 5)
    range_end = min(IQ_data.shape[1], peak_idx + 5)
    selected_bins = np.arange(range_start, range_end + 1)

    # Low-pass filtering
    D = 100
    tau_iq = 0.04
    f_low = 0.2
    downsampled = IQ_data[:, selected_bins, ::D]
    alpha_iq = np.exp(-2 / (tau_iq * original_fs))
    filtered = np.zeros_like(downsampled)
    filtered[:, :, 0] = downsampled[:, :, 0]
    for s in range(1, downsampled.shape[2]):
        filtered[:, :, s] = alpha_iq * filtered[:, :, s - 1] + (1 - alpha_iq) * downsampled[:, :, s]

    # Phase extraction
    alpha_phi = np.exp(-2 * f_low / original_fs)
    phi = np.zeros(filtered.shape[2])
    for s in range(1, filtered.shape[2]):
        z = np.sum(filtered[:, :, s] * np.conj(filtered[:, :, s - 1]))
        phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

    # Resampling signal
    t_original = np.linspace(0, original_duration, len(phi), endpoint=False)
    interp_func = interp1d(t_original, phi, kind='cubic', fill_value="extrapolate")
    new_samples = int(duration_desired * fs_desired)
    t_new = np.linspace(0, duration_desired, new_samples, endpoint=False)
    phi_resampled = interp_func(t_new)

    # Hankel Dynamic Mode Decomposition
    fs = fs_desired
    N = len(phi_resampled)
    m = 500
    H = hankel(phi_resampled[:m], phi_resampled[m - 1:])
    X, Y = H[:, :-1], H[:, 1:]

    start = time.time()
    modes, evals, freqs, _ = hankel_dmd(X, Y, fs)
    hdmd_time = time.time() - start

    mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.8)
    idx = np.where(mask)[0]
    if len(idx) > 0:
        dominant_idx = idx[np.argmax(np.abs(evals[idx]))]
        br_hz = np.abs(freqs[dominant_idx])
    else:
        br_hz = 0.3

    br_bpm = br_hz * 60
    return br_bpm, hdmd_time, phi_resampled, t_new


def main():
    parser = argparse.ArgumentParser(description="HDMD-based Breathing Rate Estimation")
    parser.add_argument("--folder", type=str, required=True, help="Folder path containing .h5 files")
    parser.add_argument("--output", type=str, default="hdmd_results.csv", help="Output CSV path")
    parser.add_argument("--save_waveform", action="store_true", help="Save extracted phase waveform per file")
    args = parser.parse_args()

    results = []
    files = sorted([f for f in os.listdir(args.folder) if f.endswith(".h5")])

    for i, fname in enumerate(files, 1):
        path = os.path.join(args.folder, fname)
        print(f"\nProcessing {fname}...")
        try:
            bpm, t_elapsed, phi, t = process_file(path)
            results.append({
                "file_index": i,
                "file_name": fname,
                "file_path": path,
                "hdmd_bpm": round(bpm, 3),
                "hdmd_time": round(t_elapsed, 3)
            })
            if args.save_waveform:
                np.save(os.path.join(args.folder, f"{fname}_phase.npy"), phi)
            print(f"✓ {fname}: {bpm:.2f} BPM in {t_elapsed:.2f}s")
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "file_index": i,
                "file_name": fname,
                "file_path": path,
                "hdmd_bpm": None,
                "hdmd_time": None,
                "error": str(e)
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
