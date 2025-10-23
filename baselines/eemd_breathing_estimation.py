#!/usr/bin/env python3
"""
EEMD baseline breathing estimation
Usage:
    python eemd_breathing_estimation.py --file path/to/h5
"""

import argparse
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.signal import welch
import time
import os
from PyEMD import EEMD


def load_and_preprocess(file_path, fs_desired=300, duration_desired=60):
    """Load radar IQ data and extract phase signal for respiration analysis."""
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

    # Resample phase signal
    t_old = np.linspace(0, original_duration, len(phi), endpoint=False)
    interp_func = interp1d(t_old, phi, kind='cubic', fill_value="extrapolate")
    new_N = int(duration_desired * fs_desired)
    t_new = np.linspace(0, duration_desired, new_N, endpoint=False)
    phi_resampled = interp_func(t_new)

    return phi_resampled, fs_desired


def analyze_breathing_eemd(signal, fs, noise_width=0.2, trials=100):
    """EEMD-based breathing rate estimation."""
    eemd = EEMD()
    eemd.noise_width = noise_width
    eemd.trials = trials
    eemd.noise_seed = 123

    imfs = eemd.eemd(signal)
    breathing_range = (0.1, 0.8)  
    imf_freqs, imf_powers = [], []

    # Analyze IMFs in frequency domain
    for imf in imfs:
        freqs, psd = welch(imf, fs=fs, nperseg=max(256, len(imf)//2))
        if len(freqs) == 0:
            imf_freqs.append(0)
            imf_powers.append(0)
            continue
        dominant_freq = freqs[np.argmax(psd)]
        mask = (freqs >= breathing_range[0]) & (freqs <= breathing_range[1])
        power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0
        imf_freqs.append(dominant_freq)
        imf_powers.append(power)

    imf_freqs = np.array(imf_freqs)
    imf_powers = np.array(imf_powers)
    target_freq = 0.3
    score = imf_powers * np.exp(-((imf_freqs - target_freq) ** 2) / 0.1)
    best_idx = np.argmax(score)

    respiration_signal = imfs[best_idx]
    freqs, psd = welch(respiration_signal, fs=fs, nperseg=max(256, len(respiration_signal)//2))
    dominant_freq = freqs[np.argmax(psd)]
    if not (0.1 <= dominant_freq <= 0.8):
        dominant_freq = 0.3

    bpm = dominant_freq * 60.0
    return bpm, respiration_signal


def estimate_breathing_eemd(file_path, fs_desired=300):
    """Main pipeline wrapper for EEMD baseline."""
    phi, fs = load_and_preprocess(file_path, fs_desired=fs_desired)
    start = time.time()
    bpm, reconstructed = analyze_breathing_eemd(phi, fs)
    elapsed = time.time() - start
    return bpm, reconstructed, elapsed


def main():
    parser = argparse.ArgumentParser(description="EEMD Baseline breathing estimation")
    parser.add_argument("--file", required=True, help="Path to .h5 radar file")
    parser.add_argument("--fs", type=float, default=300.0, help="Desired resampling rate")
    parser.add_argument("--noise", type=float, default=0.2, help="EEMD noise width")
    parser.add_argument("--trials", type=int, default=100, help="EEMD number of trials")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    print(f"Processing {args.file} with EEMD (noise={args.noise}, trials={args.trials})...")
    bpm, recon, elapsed = estimate_breathing_eemd(args.file, fs_desired=args.fs)
    print(f"EEMD Breathing Rate: {bpm:.2f} BPM | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
