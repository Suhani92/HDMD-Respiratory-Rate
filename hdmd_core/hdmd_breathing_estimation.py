#!/usr/bin/env python3
"""
HDMD Breathing Rate Estimation
===============================

This script estimates breathing rate from radar IQ data using Hankel Dynamic Mode Decomposition (HDMD). 
It supports real radar data (.h5 files) or synthetic signals for baseline comparison.

Usage:
------
# Run HDMD on a folder of radar .h5 files
python3 hdmd_breathing_estimation.py --folder /path/to/radar_data --output results.csv

# Run on a single radar file
python3 hdmd_breathing_estimation.py --file /path/to/file.h5

# Run on synthetic signal
python3 hdmd_breathing_estimation.py --synthetic synthetic_signal.npz
"""

import argparse
import numpy as np
import h5py
import pandas as pd
import os
import time
from scipy.linalg import hankel, svd, eig
from scipy.interpolate import interp1d


def hankel_dmd(X, Y, fs, rank=None, tol=1e-10):
    """Perform Hankel-DMD decomposition."""
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
    return modes, evals, frequencies


def diagonal_averaging(H):
    """Reconstruct signal from Hankel matrix."""
    m, n = H.shape
    L = m + n - 1
    result = np.zeros(L)
    for k in range(L):
        values = []
        for i in range(max(0, k - n + 1), min(m, k + 1)):
            j = k - i
            if 0 <= j < n:
                values.append(H[i, j])
        if values:
            result[k] = np.mean(values)
    return result



def preprocess_radar(file_path, fs_desired=300, duration_desired=60):
    """Load radar .h5 file and extract phase signal."""
    with h5py.File(file_path, "r") as f:
        frame = f["sessions/session_0/group_0/entry_0/result/frame"]
        real = np.array(frame["real"], dtype=np.float64)
        imag = np.array(frame["imag"], dtype=np.float64)

    IQ_data = (real + 1j * imag).transpose(2, 1, 0)
    num_sweeps = IQ_data.shape[2]
    original_fs = num_sweeps / 60.0  # 60s duration

    # Range bin selection
    magnitude = np.abs(IQ_data)
    mean_mag = np.mean(magnitude, axis=(0, 2))
    peak = np.argmax(mean_mag)
    r0, r1 = max(0, peak - 5), min(IQ_data.shape[1], peak + 5)
    selected = np.arange(r0, r1 + 1)

    # Temporal filtering
    tau_iq = 0.04
    alpha_iq = np.exp(-2 / (tau_iq * original_fs))
    filtered = np.zeros_like(IQ_data[:, selected, :])
    filtered[:, :, 0] = IQ_data[:, selected, 0]
    for s in range(1, filtered.shape[2]):
        filtered[:, :, s] = alpha_iq * filtered[:, :, s - 1] + (1 - alpha_iq) * IQ_data[:, selected, s]

    # Phase extraction
    f_low = 0.2
    alpha_phi = np.exp(-2 * f_low / original_fs)
    phi = np.zeros(filtered.shape[2])
    for s in range(1, filtered.shape[2]):
        z = np.sum(filtered[:, :, s] * np.conj(filtered[:, :, s - 1]))
        phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

    # Resample
    t_old = np.linspace(0, 60, len(phi), endpoint=False)
    interp_func = interp1d(t_old, phi, kind="cubic", fill_value="extrapolate")
    N_new = int(duration_desired * fs_desired)
    t_new = np.linspace(0, duration_desired, N_new, endpoint=False)
    phi_resampled = interp_func(t_new)

    return phi_resampled, fs_desired


def estimate_breathing(signal, fs=300, m=500):
    """Estimate breathing rate using HDMD."""
    H = hankel(signal[:m], signal[m - 1:])
    X, Y = H[:, :-1], H[:, 1:]

    start = time.time()
    _, evals, freqs = hankel_dmd(X, Y, fs)
    elapsed = time.time() - start

    mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.8)
    idx = np.where(mask)[0]
    if len(idx) > 0:
        dominant_idx = idx[np.argmax(np.abs(evals[idx]))]
        br_hz = np.abs(freqs[dominant_idx])
    else:
        br_hz = 0.3

    return br_hz * 60, elapsed


def main():
    parser = argparse.ArgumentParser(description="HDMD-based Breathing Rate Estimation")
    parser.add_argument("--file", type=str, help="Path to single .h5 radar file")
    parser.add_argument("--folder", type=str, help="Path to folder of .h5 radar files")
    parser.add_argument("--synthetic", type=str, help="Path to synthetic .npz signal file")
    parser.add_argument("--output", type=str, default="hdmd_results.csv", help="Output CSV file")
    parser.add_argument("--fs", type=float, default=300, help="Sampling frequency (Hz)")
    parser.add_argument("--m", type=int, default=500, help="Hankel window size")
    args = parser.parse_args()

    results = []

    if args.synthetic:
        data = np.load(args.synthetic)
        phi = data["phi"]
        print(f"Processing synthetic data: {args.synthetic}")
        bpm, t_elapsed = estimate_breathing(phi, fs=args.fs, m=args.m)
        results.append({"file": args.synthetic, "hdmd_bpm": bpm, "hdmd_time": t_elapsed})

    elif args.file:
        print(f"Processing radar file: {args.file}")
        phi, fs = preprocess_radar(args.file, fs_desired=args.fs)
        bpm, t_elapsed = estimate_breathing(phi, fs=fs, m=args.m)
        results.append({"file": args.file, "hdmd_bpm": bpm, "hdmd_time": t_elapsed})

    elif args.folder:
        files = sorted([f for f in os.listdir(args.folder) if f.endswith(".h5")])
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(args.folder, fname)
            print(f"\n[{i}/{len(files)}] Processing {fname}...")
            try:
                phi, fs = preprocess_radar(fpath, fs_desired=args.fs)
                bpm, t_elapsed = estimate_breathing(phi, fs=fs, m=args.m)
                results.append({
                    "file": fname,
                    "file_path": fpath,
                    "hdmd_bpm": bpm,
                    "hdmd_time": t_elapsed
                })
                print(f"✓ {fname}: {bpm:.2f} BPM ({t_elapsed:.2f}s)")
            except Exception as e:
                print(f"✗ Error in {fname}: {e}")
                results.append({"file": fname, "error": str(e)})

    else:
        print("Error: Please provide either --file, --folder, or --synthetic")
        return

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
