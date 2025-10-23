#!/usr/bin/env python3
"""
dmdt_tracking.py

Perform sliding-window DMD-t tracking on either synthetic or radar data.

Usage examples:
  # Synthetic (generate if no .npz present)
  python dmdt_tracking.py --mode synthetic --data_path synthetic_signal.npz --save_dir results/synth

  # Radar (load .npy/.csv/.mat)
  python dmdt_tracking.py --mode radar --data_path data/subject1.npy --fs 100 --save_dir results/radar
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig
import scipy.io as sio
import os

try:
    from synthetic_signal_generation import generate_synthetic_signal
except Exception:
    def generate_synthetic_signal(fs=120, duration=120, noise_std=0.5, seed=42):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        phi = 1.0 * np.sin(2 * np.pi * 0.25 * t)  
        np.random.seed(seed)
        phi += noise_std * np.random.randn(len(t))
        breathing_true = phi.copy()
        return t, phi, breathing_true


def load_data(data_path, data_type, fs_arg, duration_arg, noise_std):
    """
    Load data depending on mode.
    - synthetic: load .npz (expects t, phi, breathing_true) or generate if file not present
    - radar: load .npy/.csv/.mat as a 1D signal
    Returns: t, phi, fs
    """
    if data_type == "synthetic":
        if data_path and os.path.exists(data_path):
            data = np.load(data_path)
            t = data.get("t", None)
            phi = data.get("phi", None)
            breathing_true = data.get("breathing_true", None)
            if t is None or phi is None:
                raise ValueError(f"{data_path} does not contain 't' and 'phi'.")
            fs = 1.0 / (t[1] - t[0]) if len(t) > 1 else fs_arg
            return t, phi, fs
        else:
            t, phi, breathing_true = generate_synthetic_signal(fs=fs_arg, duration=duration_arg, noise_std=noise_std)
            fs = fs_arg
            return t, phi, fs

    elif data_type == "radar":
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError("Radar data file not found. Provide --data_path to an existing file.")
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".npy":
            phi = np.load(data_path)
        elif ext == ".csv":
            phi = np.loadtxt(data_path, delimiter=",")
        elif ext == ".mat":
            mat = sio.loadmat(data_path)
            if "phi" in mat:
                phi = np.squeeze(mat["phi"])
            elif "signal" in mat:
                phi = np.squeeze(mat["signal"])
            elif "x" in mat:
                phi = np.squeeze(mat["x"])
            else:
                raise KeyError("No recognized 1D variable (phi/signal/x) in .mat file.")
        else:
            raise ValueError("Unsupported radar file type. Use .npy, .csv, or .mat")
        fs = float(fs_arg)
        t = np.arange(len(phi)) / fs
        return t, phi, fs

    else:
        raise ValueError("data_type must be 'synthetic' or 'radar'")


def hankel_dmd(X, Y, fs, rank=None):
    """
    Perform DMD on Hankel matrices X and Y and return frequencies and amplitudes.
    """
    U, s, Vh = svd(X, full_matrices=False)
    if rank is None:
        if np.sum(s) == 0:
            r = 1
        else:
            cumsum = np.cumsum(s) / np.sum(s)
            idx = np.where(cumsum >= 0.99)[0]
            r = int(idx[0] + 1) if idx.size > 0 else min(len(s), min(X.shape) - 1)
    else:
        r = int(min(rank, len(s), min(X.shape) - 1))

    if r <= 0:
        r = 1

    U_r = U[:, :r]
    s_r = s[:r]
    Vh_r = Vh[:r, :]

    A_tilde = U_r.T @ Y @ Vh_r.T @ np.diag(1.0 / s_r)
    eigvals, _ = eig(A_tilde)

    # Convert eigenvalues to frequencies (Hz)
    # omega = log(lambda)/dt => freq (Hz) = imag(log(lambda)) / (2*pi*dt)
    dt = 1.0 / fs
    with np.errstate(all="ignore"):
        omega = np.log(eigvals) / dt
    freqs = np.imag(omega) / (2.0 * np.pi)
    amps = np.abs(eigvals)
    return freqs, amps


def dmdt_tracking(signal, fs, window_size=5.0, step_size=1.0, rank=None):
    """
    Sliding-window Hankel-DMD (DMD-t) tracking.
    Returns time_centers (s), dominant_freqs_bpm (BPM)
    """
    n = len(signal)
    w = int(round(window_size * fs))
    s = int(round(step_size * fs))
    if w <= 2:
        raise ValueError("window_size too small for the sampling rate.")

    time_centers = []
    dom_freqs = []

    for start in range(0, n - w + 1, s):
        window_signal = signal[start : start + w]
        m = min(150, w // 2)
        if m < 2:
            continue
        # Construct Hankel matrix
        H = hankel(window_signal[:m], window_signal[m - 1 :])
        if H.shape[1] < 2:
            continue
        X = H[:, :-1]
        Y = H[:, 1:]
        try:
            freqs, amps = hankel_dmd(X, Y, fs, rank=rank)
        except Exception:
            continue

        # respiratory band mask (Hz)
        mask = (np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.8)
        if np.any(mask):
            selected = np.where(mask)[0]
            idx = selected[np.argmax(amps[selected])]
            dom_freq = freqs[idx]
            time_centers.append((start + w / 2.0) / fs)
            dom_freqs.append(dom_freq * 60.0)  

    if len(time_centers) == 0:
        return np.array([]), np.array([])

    return np.array(time_centers), np.array(dom_freqs)


def plot_dmdt_tracking(t, phi, tp, bpm, save_path=None):
    """Plot normalized signal with estimated BPM overlaid (BPM on right axis)."""
    fig, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(t, phi / (np.max(np.abs(phi)) + 1e-12), color="black", alpha=0.5, label="Normalized signal")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Normalized signal")
    ax1.grid(False)

    ax2 = ax1.twinx()
    if len(tp) > 0 and len(bpm) > 0:
        ax2.plot(tp, bpm, "-o", color="tab:blue", markersize=4, linewidth=1, label="DMD-t BPM")
        ax2.set_ylabel("Breathing rate [BPM]")
        ax2.set_ylim([max(0, np.nanmin(bpm) - 5), np.nanmax(bpm) + 5])
    else:
        ax2.set_ylabel("Breathing rate [BPM]")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title("DMD-t Tracking")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="DMD-t Tracking (Hankel-DMD sliding windows)")
    parser.add_argument("--mode", choices=["synthetic", "radar"], default="synthetic",
                        help="Data mode: 'synthetic' (generate/load .npz) or 'radar' (load .npy/.csv/.mat)")
    parser.add_argument("--data_path", type=str, default="synthetic_signal.npz",
                        help="Path to .npz (synthetic) or radar file (.npy/.csv/.mat)")
    parser.add_argument("--fs", type=float, default=120.0, help="Sampling frequency for synthetic or radar (Hz)")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration for synthetic generation (s)")
    parser.add_argument("--noise_std", type=float, default=0.5, help="Noise std for synthetic generation")
    parser.add_argument("--window_size", type=float, default=5.0, help="Sliding window size (s)")
    parser.add_argument("--step_size", type=float, default=1.0, help="Step size between windows (s)")
    parser.add_argument("--rank", type=int, default=None, help="Optional truncation rank for DMD")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    t, phi, fs = load_data(args.data_path, args.mode, args.fs, args.duration, args.noise_std)
    print("[INFO] Running DMD-t tracking...")
    tp, bpm = dmdt_tracking(phi, fs, window_size=args.window_size, step_size=args.step_size, rank=args.rank)


    np.save(os.path.join(args.save_dir, "dmdt_time_points.npy"), tp)
    np.save(os.path.join(args.save_dir, "dmdt_bpm.npy"), bpm)
    print(f"[SAVED] Numeric results to: {args.save_dir}")
    plot_file = os.path.join(args.save_dir, "dmdt_tracking.png")
    plot_dmdt_tracking(t, phi, tp, bpm, save_path=plot_file)


if __name__ == "__main__":
    main()
