# Hankel Dynamic Mode Decomposition for Radar-Based Respiratory Sensing and Tracking

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://your-paper-link.com)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://yourusername.github.io/HDMD-Respiratory-Rate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Official implementation of **"Hankel Dynamic Mode Decomposition for Radar-Based Respiratory Sensing and Tracking"** accepted at NeurIPS 2025 Workshop on Learning to Sense.
> **Authors:** M Gopal Krishna*, Suhani Grover*, Chhavi Dhiman  
> **Affiliation:** Delhi Technological University  
> *Equal contribution


---

## Overview

This repository provides a robust framework for contactless respiratory monitoring using low-power mmWave radar and Hankel Dynamic Mode Decomposition (HDMD). Our method achieves **6.00% NRMSE indoors** and **1.33% NRMSE outdoors**, significantly outperforming baseline methods (VMD, EEMD, DWT) while maintaining robustness under random body movements.

![HDMD Pipeline](assets/images/pipeline_overview.png)
*Figure 1: Overview of the HDMD pipeline for respiratory sensing. The system processes raw radar IQ data through phase extraction, Hankel embedding, DMD decomposition, and respiratory mode reconstruction.*

### Key Features

- Variation trend phase extraction without unwrapping artifacts
- Koopman operator-based modal decomposition for robust signal isolation
- Real-time respiratory rate variability tracking with DMD-t
- Privacy-preserving, lighting-independent sensing
- Validated across diverse indoor/outdoor environments

---

## Dataset

Our dataset contains recordings from 24 subjects across:
- 3 postures: sitting, standing, lying
- 2 environments: indoor and outdoor
- 3 antenna configurations: no lens, plano-convex lens, Fresnel Zone Plate

**To access the dataset:**
1. Fill out the [data request form](https://forms.gle/6PBNuKAtrfQfkXsC8)
2. Download from the provided Google Drive link
3. Extract files to the `data/` directory

---

## Installation

### Requirements
- Python 3.8 or higher
- Acconeer XM125 radar module with A121 sensor
- USB cable for radar connection

### Setup
```bash
# Clone the repository
git clone https://github.com/Suhani92/HDMD-Respiratory-Rate.git
cd HDMD-Respiratory-Rate

# (Optional) Create conda environment
conda create -n hdmd python=3.8
conda activate hdmd

# Install Python dependencies
pip install -r requirements.txt

# Install Acconeer SDK
pip install --upgrade acconeer-exptool[app]>=4.9.0
```

**Note:** On some systems, you may need to use `pip3` instead of `pip`, or `python -m pip` for installation.

---

## Hardware Setup

### Initial Configuration

1. **Connect the radar module** via USB to your computer

2. **Install drivers** (if needed):
   - Windows: Driver should install automatically. If not, see [Acconeer documentation](https://docs.acconeer.com)
   - Linux: No driver needed, but you may need to add user to dialout group:
```bash
     sudo usermod -a -G dialout $USER
     # Log out and log back in for changes to take effect
```

3. **Run the setup wizard**:
```bash
   python -m acconeer.exptool.setup
```

4. **Launch Acconeer Exploration Tool**:
```bash
   python -m acconeer.exptool.app
```

5. **Configure connection**:
   - Select **A121** as sensor type
   - Set port type to **Serial**
   - Select port **XE125** (typically `/dev/ttyUSB0` on Linux or `COM#` on Windows)
   - Click **Connect**

### Flash the Radar (First-Time Setup Only)

If this is your first time using the radar module:

1. Navigate to the **Flash** tab in Exploration Tool
2. Select **Get latest binary for A121**
3. Put the radar in bootloader mode:
   - Press and hold the **DFU** button
   - Press the **RESET** button (while holding DFU)
   - Release **RESET**
   - Release **DFU**
4. Follow on-screen instructions to complete flashing
5. Press and release the **RESET** button after flashing completes

### Radar Configuration Parameters

Navigate to the **Stream** tab and select **Sparse IQ** service. Configure the following:

**Metadata:**
- Frame data length: `40`
- Max sweep rate: `2575 Hz`

**Processor Parameters:**
- Amplitude method: `Coherent`

**Sensor Parameters:**
- Sweeps per frame: `1`

**Subsweep Parameters:**
- Start point: `80`
- Number of points: `40`
- Step length: `8`
- HWAAS: `8`
- Profile: `3`
- Receiver gain: `16`
- PRF: `15.6 MHz`
- Enable transmitter: `✓`

**Recording:**
- Click **Start measurement** to begin data collection
- Save data as `.h5` file for processing

---

## Code Structure
```
HDMD-Respiratory-Rate/
├── Variation_Trend_Phase_Extraction.py    # Phase signal extraction from radar data
├── phase-allalgos.py                      # Single file decomposition (all methods)
├── phase-allalgosICU-loop.py             # Batch processing for indoor dataset
├── phase-allalgos-outdoor-loop.py        # Batch processing for outdoor dataset
├── phase-allalgos-lens-loop.py           # Lens configuration comparison
├── DMDtracking_synthetic_signal.py       # DMD-t tracking demonstration
└── README.md                              # This file
```

### Code Descriptions

#### 1. `Variation_Trend_Phase_Extraction.py`
Extracts continuous phase signal from raw radar IQ data using the variation trend method. This approach avoids phase unwrapping artifacts by computing cumulative displacement over time.

**Key functions:**
- Loads radar data from `.h5` files
- Performs range bin selection and temporal filtering
- Extracts continuous phase using complex conjugation
- Resamples to desired frequency (default: 300 Hz)

**Usage:**
```bash
python Variation_Trend_Phase_Extraction.py
```

Edit the `file_path` variable to point to your radar data file.

#### 2. `phase-allalgos.py`
Comprehensive comparison of HDMD against baseline methods (EEMD, VMD, DWT) for a single data file. Generates reconstructed respiratory waveforms and breathing rate estimates.

**Methods implemented:**
- Hankel Dynamic Mode Decomposition (HDMD)
- Ensemble Empirical Mode Decomposition (EEMD)
- Variational Mode Decomposition (VMD)
- Discrete Wavelet Transform (DWT)

**Usage:**
```bash
python phase-allalgos.py
```

Outputs 5 comparison plots showing original signal and reconstructions from each method.

#### 3. `phase-allalgosICU-loop.py`
Batch processing script for indoor dataset. Processes all files (i1.h5 to i24.h5) and generates CSV with breathing rate estimates and computational times.

**Usage:**
```bash
python phase-allalgosICU-loop.py
```

**Outputs:**
- `breathing_rates_results.csv`: Contains BPM estimates and processing times for all methods
- Console output with per-file results and summary statistics

#### 4. `phase-allalgos-outdoor-loop.py`
Batch processing script for outdoor dataset. Similar to indoor processing but optimized for outdoor signal characteristics.

**Usage:**
```bash
python phase-allalgos-outdoor-loop.py
```

Update `folder_path` variable to point to your outdoor data directory.

#### 5. `phase-allalgos-lens-loop.py`
Compares respiratory rate estimation across three antenna configurations:
- No lens
- Plano-convex lens
- Fresnel Zone Plate lens

**Usage:**
```bash
python phase-allalgos-lens-loop.py
```

Generates comparative analysis showing how antenna configuration affects signal quality and estimation accuracy.

#### 6. `DMDtracking_synthetic_signal.py`
Demonstrates DMD-t tracking on synthetic respiratory signal with time-varying breathing rates. Creates side-by-side comparison with STFT spectrogram.

**Usage:**
```bash
python DMDtracking_synthetic_signal.py
```

**Outputs:**
- `dmd_spectrogram.svg`: DMD-t tracking vs STFT comparison
- `signal.svg`: Synthetic signal with rate transitions

---

## Method Overview

![HDMD Algorithm](assets/images/algorithm_flowchart.png)
*Figure 2: HDMD algorithm flowchart showing the complete processing pipeline from raw IQ data to respiratory rate estimation.*

### Phase Extraction

The variation trend method computes continuous phase by tracking cumulative displacement:
```
φ[s] = α_φ · φ[s-1] + ∠(Σ x̄[s,d'] · x̄*[s-1,d'])
```

where `α_φ` acts as a high-pass filter to suppress low-frequency drift.

### Hankel-DMD Decomposition

1. **Hankel Matrix Construction**: Time-delayed embeddings create pseudo-spatiotemporal representation
2. **SVD-based DMD**: Approximates Koopman operator via `Y ≈ KX`
3. **Mode Selection**: Isolates respiratory modes (0.1-0.8 Hz range)
4. **Reconstruction**: Combines selected modes to recover breathing waveform

### DMD-t Tracking

![DMD-t Tracking Results](assets/images/tracking_comparison.png)
*Figure 3: DMD-t provides superior time-frequency resolution compared to STFT, enabling accurate tracking of respiratory rate variations.*

Sliding window DMD enables real-time tracking of time-varying respiratory rates. Each window produces local eigenvalues representing instantaneous breathing frequency.

---

## Results

### Performance Comparison

| Method | Indoor RMSE (BPM) | Indoor NRMSE (%) | Outdoor RMSE (BPM) | Outdoor NRMSE (%) |
|--------|-------------------|------------------|---------------------|-------------------|
| **HDMD** | **1.14** | **6.00** | **0.40** | **1.33** |
| VMD | 6.12 | 32.20 | 8.56 | 28.55 |
| EEMD | 4.68 | 24.60 | 3.81 | 12.71 |
| DWT | 9.27 | 48.80 | 13.40 | 44.66 |

![Ground Truth Comparison](assets/images/ground_truth_comparison.png)
*Figure 4: Comparison of estimated vs. ground truth breathing rates across 24 subjects in indoor conditions.*

### Computational Efficiency

| Method | Average Time (s) | Relative Speed |
|--------|-----------------|----------------|
| DWT | 0.004 | 1350× faster |
| VMD | 0.86 | 6.3× faster |
| **HDMD** | **5.4** | baseline |
| EEMD | 38.9 | 7.2× slower |

While HDMD has higher computational cost, it provides significantly better accuracy and robustness, particularly under challenging conditions with body movements.

---

## Troubleshooting

### Common Issues

**Radar not detected:**
- Ensure USB cable is properly connected
- Check that drivers are installed (Windows)
- Verify device appears in device manager/lsusb output
- Try a different USB port

**Phase extraction fails:**
- Verify `.h5` file structure matches expected format
- Check that radar was configured with correct parameters
- Ensure adequate signal quality (subject within 1.5-2m range)

**Poor breathing rate estimates:**
- Confirm subject is within optimal range (1.5-2m)
- Check for environmental interference or multipath
- Verify antenna lens is properly installed
- Try different posture if results are inconsistent

**Memory errors during processing:**
- Reduce embedding dimension `m` in Hankel matrix construction
- Process shorter signal segments
- Close other applications to free memory


---

## Contact

- M Gopal Krishna: mgopalkrishna0007@gmail.com
- Suhani Grover: suhani1077@gmail.com
- Chhavi Dhiman: chhavi.dhiman@dtu.ac.in

---

## License

This project is released under the MIT License. See `LICENSE` file for details.

**Note:** This implementation is intended for research purposes. Clinical applications require appropriate validation and regulatory compliance.

---
