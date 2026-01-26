#!/usr/bin/env python3
"""
DWT baseline breathing estimation (wavelet + FFT selection)
Usage:
    python dwt_breathing_estimation.py --file path/to/h5
"""
import argparse
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.signal import welch
import pywt
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
    r1 = min(IQ_data.shape[1] - 1, peak_idx[0] + 5)  
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


def analyze_wavelet_breathing(signal, fs, wavelet='db4', max_level=4):
    optimal_level = min(max_level, int(np.floor(np.log2(len(signal)))))
    coeffs = pywt.wavedec(signal, wavelet, level=optimal_level, mode='periodization')
    coeff_names = [f'A{optimal_level}'] + [f'D{i}' for i in range(optimal_level, 0, -1)]

    best_component = None
    best_rate = 0.0
    best_name = None
    max_resp_power = -np.inf

    for i, (c, name) in enumerate(zip(coeffs, coeff_names)):
        # reconstruct only the i-th component
        test_coeffs = [np.zeros_like(cc) if idx != i else cc for idx, cc in enumerate(coeffs)]
        recon = pywt.waverec(test_coeffs, wavelet, mode='periodization')

        # trim/pad
        if len(recon) > len(signal):
            recon = recon[:len(signal)]
        elif len(recon) < len(signal):
            recon = np.pad(recon, (0, len(signal) - len(recon)))

        # spectral analysis
        fft_vals = np.fft.fft(recon)
        fft_freqs = np.fft.fftfreq(len(recon), d=1.0/fs)
        pos_mask = (fft_freqs >= 0)
        pos_freqs = fft_freqs[pos_mask]
        pos_mag = np.abs(fft_vals[pos_mask])

        resp_mask = (pos_freqs >= 0.04) & (pos_freqs <= 0.6)  
        if not np.any(resp_mask):
            continue

        masked_mag = pos_mag[resp_mask]
        masked_freqs = pos_freqs[resp_mask]
        dominant_idx = np.argmax(masked_mag)
        dominant_freq = masked_freqs[dominant_idx]
        resp_power = masked_mag[dominant_idx]

        if resp_power > max_resp_power:
            max_resp_power = resp_power
            best_component = recon
            best_rate = dominant_freq
            best_name = name

    if best_component is None:
        a_idx = 0
        test_coeffs = [np.zeros_like(cc) if idx != a_idx else cc for idx, cc in enumerate(coeffs)]
        best_component = pywt.waverec(test_coeffs, wavelet, mode='periodization')[:len(signal)]
        fft_vals = np.fft.fft(best_component)            # spectral estimate
        fft_freqs = np.fft.fftfreq(len(best_component), d=1.0/fs)
        pos_mask = (fft_freqs >= 0)
        pos_freqs = fft_freqs[pos_mask]
        pos_mag = np.abs(fft_vals[pos_mask])
        resp_mask = (pos_freqs >= 0.04) & (pos_freqs <= 0.6)
        if np.any(resp_mask):
            best_rate = pos_freqs[resp_mask][np.argmax(pos_mag[resp_mask])]
        else:
            best_rate = 0.3

    return best_component, best_name, best_rate


def estimate_breathing_dwt(file_path, wavelet='db4', max_level=4, fs_desired=300):
    phi, fs = load_and_preprocess(file_path, fs_desired=fs_desired)
    start = time.time()
    comp, name, freq_hz = analyze_wavelet_breathing(phi, fs, wavelet=wavelet, max_level=max_level)
    elapsed = time.time() - start
    bpm = freq_hz * 60.0
    return bpm, comp, elapsed


def main():
    parser = argparse.ArgumentParser(description="DWT Baseline breathing estimation")
    parser.add_argument("--file", required=True, help="Path to .h5 radar file")
    parser.add_argument("--wavelet", default='db4', help="Wavelet name")
    parser.add_argument("--max_level", type=int, default=4, help="Max decomposition level")
    parser.add_argument("--fs", type=float, default=300.0, help="Desired resampling rate")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    print(f"Processing {args.file} with DWT (wavelet={args.wavelet})...")
    bpm, comp, elapsed = estimate_breathing_dwt(args.file, wavelet=args.wavelet, max_level=args.max_level, fs_desired=args.fs)
    print(f"DWT Breathing Rate: {bpm:.2f} BPM | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
