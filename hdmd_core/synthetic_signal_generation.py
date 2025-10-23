# !/usr/bin/env python3
"""
synthetic_signal_generation.py
Generate a synthetic respiratory + cardiac signal with noise.
"""

import numpy as np

def generate_synthetic_signal(fs=120, duration=120, noise_std=0.5, seed=42):
    """
    Parameters:
        fs (int): Sampling frequency in Hz
        duration (float): Duration in seconds
        noise_std (float): Standard deviation of Gaussian noise
        seed (int): Random seed for reproducibility
        
    Returns:
        t (ndarray): Time vector
        phi (ndarray): Synthetic mixed signal
        breathing_true (ndarray): True breathing component
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    phi = np.zeros_like(t)

    # Breathing intervals
    interval1 = t < 40
    interval2 = (t >= 40) & (t < 80)
    interval3 = t >= 80

    # Heart and noise
    freq_heart = 1.2  # 72 BPM
    comp_heart = np.cos(2 * np.pi * freq_heart * t)
    comp_noise = 0.005 * np.cos(2 * np.pi * 5.0 * t)

    # Segment 1 (15 BPM)
    amp1 = 5.0 + 0.2 * np.sin(2 * np.pi * 0.02 * t[interval1])
    comp1 = amp1 * np.cos(2 * np.pi * 0.25 * t[interval1])

    # Segment 2 (17 BPM)
    amp2 = 5.0 + 0.3 * np.sin(2 * np.pi * 0.015 * t[interval2])
    comp2 = amp2 * np.cos(2 * np.pi * (17/60) * (t[interval2] - 40))

    # Segment 3 (12 BPM)
    amp3 = 5.0 + 0.15 * np.sin(2 * np.pi * 0.025 * t[interval3])
    comp3 = amp3 * np.cos(2 * np.pi * (12/60) * (t[interval3] - 80))

    phi[interval1] = comp1 + comp_heart[interval1] + comp_noise[interval1]
    phi[interval2] = comp2 + comp_heart[interval2] + comp_noise[interval2]
    phi[interval3] = comp3 + comp_heart[interval3] + comp_noise[interval3]


    np.random.seed(seed)
    phi += noise_std * np.random.randn(len(t))

    breathing_true = np.zeros_like(t)
    breathing_true[interval1] = comp1
    breathing_true[interval2] = comp2
    breathing_true[interval3] = comp3

    return t, phi, breathing_true

if __name__ == "__main__":
    t, phi, breathing_true = generate_synthetic_signal()
    np.savez("synthetic_signal.npz", t=t, phi=phi, breathing_true=breathing_true)
    print("✅ Synthetic signal saved as synthetic_signal.npz")
