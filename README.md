## Solar Flare — Dynamic Visualization (Notebook)

This repo contains a small, self‑contained notebook that turns the UCI “Solar Flare” dataset into a starfield‑style animation. It cleans the data, reduces it to 2D with PCA, and renders a moving trail with a simple numeric readout at the bottom. You can export the animation as MP4 (preferred) or GIF (fallback) right from the notebook.

### What’s here
- Solar Flare.ipynb — main notebook: fetch data, build the animation, export MP4/GIF
- visualization.py — a minimal static plotting example
- simple neural network data.ipynb — toy data demo

### Setup
- Python 3.10+ recommended
- Create and use a virtual environment if you like
- Install packages:
	- pandas, numpy, matplotlib, scikit-learn, ucimlrepo, pillow
- Optional for MP4 export: install ffmpeg on your system (the notebook falls back to GIF if ffmpeg isn’t available)

Example (Ubuntu):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pandas numpy matplotlib scikit-learn ucimlrepo pillow
sudo apt-get update && sudo apt-get install -y ffmpeg   # optional, for MP4
```

### How to run
1) Open the repo in VS Code and start the Jupyter Notebook experience.
2) Select your virtual environment as the kernel.
3) Open Solar Flare.ipynb and run:
	 - Cell 1: downloads and prepares the data (defines X and y)
	 - Cell 2: builds the animation and writes the output file

The notebook prefers to save an MP4 named `solar_flare.mp4`. If ffmpeg isn’t available, it automatically saves a GIF named `solar_flare.gif` instead.

### Tuning knobs (in Cell 2)
- num_frames: total frames to render (<= 300 by default to stay memory‑friendly)
- trail_len: how many recent points to show in the moving window
- fps: 12–15 is smooth without being heavy
- dpi: 90 is a good default

### What the plot shows
- PCA(2D) view of the dataset with a faint background of all points
- A bright sliding trail that “flies” through the cloud
- Point color and size are based on the target value (encoded to numeric)
- A bottom readout shows: index, PCA coords, target, and the color/size scalars

### Troubleshooting
- No MP4 file: install ffmpeg and re‑run Cell 2, or use the GIF fallback
- Text readout missing in the video: it’s drawn on the axes with extra bottom margin; if needed, try lowering fps or turning off blit temporarily
- Kernel crash / slow render: reduce num_frames, fps, dpi, or trail_len
- “X is not defined”: run Cell 1 before Cell 2 (the data is created in Cell 1)

### Data source
UCI Machine Learning Repository — Solar Flare dataset (id=89), accessed via `ucimlrepo`.
