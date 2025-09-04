Windows CAVA-like Terminal Visualizer (Minimal)

This is a minimal CAVA-style spectrum visualizer for Windows terminals. It captures system (loopback) audio with WASAPI via `sounddevice`, computes FFTs with `numpy`, and renders animated bars using ANSI escape codes (no curses needed).

Requirements
- Python 3.9+
- Windows 10+ terminal with ANSI support (default on newer builds)
- Packages: `numpy`, and one of:
  - `sounddevice` (preferred, with WASAPI loopback support)
  - or `soundcard` (fallback WASAPI loopback backend)

Install
```bash
pip install --upgrade pip
pip install numpy sounddevice
# If sounddevice lacks WASAPI loopback on your system, install fallback:
pip install soundcard
```

Run
```bash
python cava_win.py
# If WASAPI loopback via sounddevice fails, use the fallback:
python cava_win.py --backend sc
```

Keys
- `q` or `Ctrl+C`: quit

Notes
- Uses WASAPI loopback from your default output device. If WASAPI is unavailable, the program exits with a clear message.
- If your terminal flickers, try reducing FPS or bar count via CLI flags: `--fps 30 --bars 48`.
- If you previously saw `Invalid number of channels`, this build now auto-matches the loopback device channel count. If it persists, update `sounddevice` and try `--samplerate 48000` or your device rate.

Device Selection & Debug
- List devices: `python cava_win.py --list-devices`
- Pick device by index: `python cava_win.py --device 3`
- Pick device by name: `python cava_win.py --device "speakers"`
- Debug prints for probing: `python cava_win.py --debug`
- Force channel count (try 2 or 1): `python cava_win.py --channels 2`
- Force backend: `--backend sd` (sounddevice) or `--backend sc` (soundcard)

Themes & Visual Options
- Choose theme: `--theme sunset|ocean|rainbow|mono` (default: `sunset`)
- Peak indicator: `--cap dot|line|none` (default: `dot`)
- Disable 24-bit color (use 256-color): `--no-truecolor`

Examples
```bash
# Ocean theme with line peaks
python cava_win.py --theme ocean --cap line

# Rainbow theme in 256-color mode
python cava_win.py --theme rainbow --no-truecolor

# Minimal mono theme
python cava_win.py --theme mono --cap none
```

If you still get a channel error, try combining flags, e.g.:
```bash
python cava_win.py --device "speakers" --samplerate 44100 --debug
```

If your `sounddevice` build doesn’t support `WasapiSettings(loopback=True)` (you’ll see probing with `extra=none`), try the soundcard backend:
```bash
pip install soundcard
python cava_win.py --backend sc --debug
# You can still pick a device by index or name hint:
python cava_win.py --backend sc --device 7 --debug
python cava_win.py --backend sc --device "speakers" --samplerate 48000 --debug
```

CLI Options
```text
python cava_win.py --help
```
