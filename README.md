# Virtual Choir Conductor - SATB

## Description
A keyboard-based virtual conducting system for choir music. The system automatically detects time signatures from music scores and guides the conductor through the correct pattern (2/4, 3/4, 4/4). Each correct beat triggers a SATB chord (Soprano, Alto, Tenor, Bass).

## Features
- Automatic time signature detection from score metadata (2/4, 3/4, 4/4)
- Real-time conducting pattern validation
- SATB (Soprano, Alto, Tenor, Bass) chord playback
- Real-time BPM calculation
- Visual pattern guide with next key highlighting
- 6 pre-loaded choir scores
- Multi-modal feedback (visual, auditory, textual)

## Requirements
- Python 3.10+
- Ubuntu 20.04 / Windows / macOS

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/virtual-choir-conductor.git
cd virtual-choir-conductor

# Create virtual environment
python3 -m venv choir_env
source choir_env/bin/activate  # Linux/Mac
# choir_env\Scripts\activate  # Windows

# Install dependencies
pip install opencv-python numpy simpleaudio
