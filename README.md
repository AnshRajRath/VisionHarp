# VisionHarp

> A touchless, gesture-controlled virtual instrument powered by computer vision and real-time audio synthesis.

## Overview

VisionHarp turns a webcam into a musical controller. The system detects and tracks a hand in real time, maps its position to musical parameters, and synthesizes audio dynamically without requiring a physical instrument or MIDI controller.

The project combines computer vision, signal processing, smoothing, musical scale logic, and stereo audio generation into one interactive pipeline.

## Key features

- Touchless hand-controlled music generation
- HSV + YCrCb skin detection
- Palm-center tracking using distance transform
- Kalman filtering + exponential moving average smoothing
- Note hysteresis to reduce unstable note switching
- Multiple musical scales
- Real-time sine / harmonic synthesis
- Vibrato through an LFO
- Stereo constant-power panning
- Feedback delay / echo effects
- Webcam-based interactive UI

## Tech stack

- Python
- OpenCV
- NumPy
- PyAudio
- Jupyter Notebook

## Signal flow

```text
Webcam frame
    ↓
Skin segmentation
    ↓
Hand contour + palm center
    ↓
Kalman + EMA smoothing
    ↓
Position → note / volume / pan / modulation
    ↓
Real-time software synthesizer
    ↓
Stereo audio output
```

## Repository structure

```text
VisionHarp/
├── prj.ipynb          # Main implementation
├── complete_desc.md   # Detailed technical / viva documentation
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/AnshRajRath/VisionHarp.git
cd VisionHarp
pip install -r requirements.txt
jupyter notebook prj.ipynb
```

A working webcam and audio output device are required.

## Interesting implementation details

### Stable hand tracking

VisionHarp combines calibrated HSV and YCrCb thresholding with morphological cleanup, palm-center estimation, Kalman tracking, and EMA smoothing. This helps reduce jitter while keeping interaction responsive.

### Musical control

Horizontal motion can select notes, while other spatial properties can control parameters such as volume, panning, or modulation. Hysteresis prevents rapid flickering between adjacent notes when the hand sits near a boundary.

### Audio synthesis

The synthesizer uses continuous phase accumulation and harmonic components to produce a richer tone, then adds modulation, stereo panning, and feedback delay for a more expressive sound.

## Detailed documentation

For the full implementation breakdown, mathematics, processing pipeline, and viva-oriented notes, see [`complete_desc.md`](complete_desc.md).

## Author

**Ansh Raj Rath**  
GitHub: [@AnshRajRath](https://github.com/AnshRajRath)
