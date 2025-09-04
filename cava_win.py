import sys
import os
import math
import time
import threading
import queue
import shutil
import argparse

# Attempt imports that require installation
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("This program requires numpy. Install with: pip install numpy", file=sys.stderr)
    raise

try:
    import sounddevice as sd
    from sounddevice import PortAudioError
except Exception as e:  # pragma: no cover
    print("This program requires sounddevice. Install with: pip install sounddevice", file=sys.stderr)
    raise

# Optional fallback backend: python-soundcard (WASAPI loopback)
try:
    import soundcard as sc  # type: ignore
    SC_AVAILABLE = True
except Exception:
    SC_AVAILABLE = False


# Enable ANSI escapes on Windows consoles (older builds)
def _enable_vt_mode():
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(h, new_mode)
    except Exception:
        pass


def find_wasapi_output_device(preferred: int | None = None, name_hint: str | None = None):
    """Find default output device on WASAPI hostapi for loopback.

    If 'preferred' index is valid and WASAPI, use it. If 'name_hint' is provided,
    try to find a device whose name contains the hint.
    """
    # Find WASAPI hostapi
    wasapi_index = None
    for i, api in enumerate(sd.query_hostapis()):
        if 'wasapi' in api.get('name', '').lower():
            wasapi_index = i
            break
    if wasapi_index is None:
        return None

    devices = sd.query_devices()

    # Preferred explicit index
    if preferred is not None and 0 <= preferred < len(devices):
        d = devices[preferred]
        if d.get('hostapi') == wasapi_index and d.get('max_output_channels', 0) > 0:
            return preferred

    # Name hint selection
    if name_hint:
        hint = name_hint.lower()
        cand = [
            (i, d) for i, d in enumerate(devices)
            if d.get('hostapi') == wasapi_index and d.get('max_output_channels', 0) > 0
            and hint in str(d.get('name', '')).lower()
        ]
        if cand:
            return cand[0][0]

    # Default output for WASAPI
    api = sd.query_hostapis(wasapi_index)
    default_out = api.get('default_output_device', -1)
    if default_out is not None and default_out >= 0:
        return default_out

    # Any output device on WASAPI
    for idx, d in enumerate(devices):
        if d.get('hostapi') == wasapi_index and d.get('max_output_channels', 0) > 0:
            return idx
    return None


def map_index_to_wasapi_output(index: int | None):
    """If index refers to a non-WASAPI output device, try to find the WASAPI
    output device with a similar name (for loopback capture)."""
    if index is None:
        return None
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return index
    wasapi_index = None
    for i, api in enumerate(hostapis):
        if 'wasapi' in api.get('name', '').lower():
            wasapi_index = i
            break
    if wasapi_index is None:
        return index
    if index < 0 or index >= len(devices):
        return index
    base = devices[index]
    if base.get('hostapi') == wasapi_index:
        return index
    base_name = str(base.get('name', '')).lower()
    # Try exact-like match first, then substring
    candidates = []
    for i, d in enumerate(devices):
        if d.get('hostapi') != wasapi_index:
            continue
        if d.get('max_output_channels', 0) <= 0:
            continue
        name = str(d.get('name', '')).lower()
        if base_name == name or base_name in name or name in base_name:
            candidates.append(i)
    return candidates[0] if candidates else index


def find_named_loopback_input_device():
    """Fallback: find an input device whose name suggests loopback capture."""
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for idx, d in enumerate(devices):
        name = str(d.get('name', '')).lower()
        if 'loopback' in name and d.get('max_input_channels', 0) > 0:
            return idx
    return None


class AudioLoopback:
    def __init__(self, samplerate=48000, blocksize=2048, channels=2, device=None, name_hint=None, debug=False, backend='auto'):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels  # requested; may be overridden to match device
        self.q = queue.Queue(maxsize=8)
        self.stream = None
        self.device = device
        self.name_hint = name_hint
        self.debug = debug
        self.backend = backend
        # soundcard backend state
        self._sc_recorder = None
        self._sc_thread = None
        self._running = False

    def start(self):
        # Backend selection
        want_sd = self.backend in ('auto', 'sd')
        want_sc = self.backend in ('auto', 'sc')

        if want_sd and hasattr(sd, 'WasapiSettings'):
            try:
                self._start_sounddevice()
                if self.debug:
                    print("[audio] backend= sounddevice (WASAPI loopback)")
                return
            except Exception as e:
                if self.debug:
                    print(f"[audio] sounddevice backend failed: {e}")
                if self.backend == 'sd':
                    raise

        if want_sc and SC_AVAILABLE:
            self._start_soundcard()
            if self.debug:
                print("[audio] backend= soundcard (WASAPI loopback)")
            return

        print("No working audio backend found. Install a newer 'sounddevice' (with WASAPI loopback) or install fallback with: pip install soundcard")
        sys.exit(1)

    def _start_sounddevice(self):
        device = find_wasapi_output_device(preferred=self.device, name_hint=self.name_hint)
        if device is None:
            print("WASAPI not available or no output device found. This tool only supports Windows WASAPI loopback.")
            sys.exit(1)

        wasapi_settings = None
        # Try constructing WasapiSettings with loopback
        ws_errors = []
        for kwargs in (
            dict(loopback=True, exclusive=False),
            dict(loopback=True),
        ):
            try:
                wasapi_settings = sd.WasapiSettings(**kwargs)
                break
            except Exception as e:
                ws_errors.append(str(e))
                wasapi_settings = None

        def callback(indata, frames, timeinfo, status):
            if status:
                # Drop on overflow/underflow; minimal logging to avoid spam.
                pass
            try:
                self.q.put_nowait(indata.copy())
            except queue.Full:
                # Drop if consumer too slow
                pass

        def try_open_with_candidates(dev_index, extra):
            di = sd.query_devices(dev_index)
            base_sr = int(self.samplerate)
            dev_sr = int(di.get('default_samplerate') or base_sr)
            # Build samplerate candidates
            sr_candidates = []
            for s in [base_sr, dev_sr, 48000, 44100]:
                if s and s not in sr_candidates:
                    sr_candidates.append(int(s))
            # Build channel candidates
            if extra is not None:
                # For WASAPI loopback, PortAudio reports max_input_channels on the OUTPUT device
                base_ch = int(di.get('max_input_channels') or di.get('max_output_channels') or 2)
            else:
                base_ch = int(di.get('max_input_channels') or 1)
            ch_candidates = []
            # Prefer device-advertised input channels first, then common fallbacks
            prefer = [self.channels, base_ch, min(2, base_ch), 2, 1]
            # Add surround sizes only if device hints they may be supported
            if base_ch >= 6:
                prefer.extend([6])
            if base_ch >= 8:
                prefer.extend([8])
            for c in prefer:
                if c and c not in ch_candidates:
                    ch_candidates.append(int(c))

            last_err = None
            for sr in sr_candidates:
                for ch in ch_candidates:
                    try:
                        if self.debug:
                            name = di.get('name', '?')
                            print(
                                f"[audio] probing device={dev_index} '{name}' sr={sr} ch={ch} extra={'loopback' if extra else 'none'} "
                                f"hostapi={di.get('hostapi')} caps(in={di.get('max_input_channels')}, out={di.get('max_output_channels')})"
                            )
                        stream = sd.InputStream(
                            samplerate=sr,
                            blocksize=self.blocksize,
                            device=dev_index,
                            channels=ch,
                            dtype='float32',
                            latency='low',
                            callback=callback,
                            extra_settings=extra,
                        )
                        # Actually start/stop to verify the configuration
                        stream.start()
                        stream.stop()
                        self.channels = ch
                        self.samplerate = sr
                        return stream
                    except (PortAudioError, ValueError) as e:
                        last_err = e
                        continue
            if last_err:
                raise last_err
            raise RuntimeError("No valid audio configuration found")

        try:
            # Primary path: WASAPI loopback using output device
            self.stream = try_open_with_candidates(device, wasapi_settings)
        except TypeError:
            # Older sounddevice may not support extra_settings
            loopback_input = find_named_loopback_input_device()
            if loopback_input is None:
                raise
            self.stream = try_open_with_candidates(loopback_input, None)
        # Ensure the selected stream is running
        if self.stream is not None:
            self.stream.start()

    def _start_soundcard(self):
        # Resolve a target speaker name (for mapping to loopback microphone)
        target_name = None
        # If numeric device index provided, map to its name as a hint
        if self.device is not None:
            try:
                dn = sd.query_devices(self.device).get('name', None)
                if dn:
                    target_name = dn
            except Exception:
                pass
        if target_name is None and self.name_hint:
            target_name = self.name_hint
        # Fallback to default speaker name
        if target_name is None:
            try:
                target_name = sc.default_speaker().name
            except Exception:
                try:
                    spkrs = sc.all_speakers()
                    target_name = spkrs[0].name if spkrs else None
                except Exception:
                    target_name = None
        if target_name is None:
            raise RuntimeError("No speaker name could be resolved for soundcard backend")

        # Obtain loopback microphone corresponding to target speaker
        mic = None
        try:
            mic = sc.get_microphone(target_name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            # Try scanning all loopback microphones by name
            try:
                mics = sc.all_microphones(include_loopback=True)
                low = target_name.lower()
                for m in mics:
                    try:
                        nm = m.name.lower()
                        if low in nm or 'loopback' in nm:
                            mic = m
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if mic is None:
            raise RuntimeError("No loopback microphone found for soundcard backend")

        ch = 2 if (self.channels is None) else max(1, int(self.channels))
        # Create a recorder from the loopback microphone
        try:
            self._sc_recorder = mic.recorder(samplerate=self.samplerate, channels=ch, blocksize=self.blocksize)
        except Exception:
            # Retry with safer defaults
            if self.samplerate != 44100:
                self.samplerate = 44100
            ch = 1
            self._sc_recorder = mic.recorder(samplerate=self.samplerate, channels=ch, blocksize=self.blocksize)
        self.channels = ch

        self._running = True
        def _loop():
            try:
                with self._sc_recorder:
                    while self._running:
                        data = self._sc_recorder.record(self.blocksize)
                        try:
                            self.q.put_nowait(data.astype('float32', copy=False))
                        except queue.Full:
                            pass
            except Exception:
                self._running = False
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        self._sc_thread = t
        if self.debug:
            try:
                print(f"[audio] soundcard using loopback mic: {mic.name} (target speaker: {target_name})")
            except Exception:
                pass


    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
            if self._sc_thread is not None:
                self._running = False
                self._sc_thread.join(timeout=0.5)
        finally:
            self.stream = None
            self._sc_thread = None


def log_space_frequencies(n_bars, f_min, f_max, samplerate, n_fft):
    # Log-spaced center frequencies
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), num=n_bars)
    # Map to fft bin indices (one-sided)
    bin_freqs = np.fft.rfftfreq(n_fft, d=1.0/samplerate)
    idxs = np.clip(np.searchsorted(bin_freqs, freqs), 1, len(bin_freqs) - 1)
    # Ensure strictly increasing indices
    idxs = np.maximum.accumulate(idxs)
    return idxs


class Visualizer:
    def __init__(self, bars=64, fps=60, height=None, fmin=43, fmax=16000, decay=0.6, rise=0.5, gain=1.0,
                 theme='sunset', cap='dot', truecolor=True):
        self.bars = bars
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.decay = decay  # 0..1, closer to 1 = slower fall
        self.rise = rise    # 0..1, closer to 1 = slower rise (more smoothing)
        self.gain = gain
        self.height = height
        self.theme = theme
        self.cap = cap  # 'dot', 'line', 'none'
        self.truecolor = truecolor
        self.prev_vals = np.zeros(self.bars, dtype=np.float32)
        self.peak_vals = np.zeros(self.bars, dtype=np.float32)
        self.peak_decay = 0.95

    def _term_size(self):
        size = shutil.get_terminal_size(fallback=(80, 24))
        width, height = size.columns, size.lines
        if self.height is not None:
            height = min(height, self.height)
        return width, height

    def _clear_and_home(self):
        # Hide cursor, clear screen, home cursor
        sys.stdout.write("\x1b[?25l\x1b[H")
        sys.stdout.flush()

    def _restore_cursor(self):
        sys.stdout.write("\x1b[?25h\x1b[0m\n")
        sys.stdout.flush()

    def _ansi_color(self, r, g, b):
        if self.truecolor:
            return f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m"
        # fallback to 256-color cube
        def _q(v):
            return max(0, min(5, int(round(v / 51))))
        ri, gi, bi = _q(r), _q(g), _q(b)
        idx = 16 + 36*ri + 6*gi + bi
        return f"\x1b[38;5;{idx}m"

    def _gradient_color(self, t, bar_pos=None, now=None):
        # t in [0,1] from bottom(0) to top(1)
        theme = (self.theme or 'sunset').lower()
        if theme == 'mono':
            v = 200 + int(55 * t)
            return self._ansi_color(v, v, v)
        elif theme == 'ocean':
            # deep blue -> cyan -> white
            stops = [
                (0.0, (0, 40, 120)),
                (0.5, (0, 170, 255)),
                (1.0, (220, 245, 255)),
            ]
        elif theme == 'sunset':
            # purple -> magenta -> orange -> gold
            stops = [
                (0.0, (75, 0, 110)),
                (0.45, (200, 0, 120)),
                (0.75, (255, 120, 0)),
                (1.0, (255, 220, 120)),
            ]
        elif theme == 'rainbow':
            # HSV rainbow across bars and rows
            # base hue cycles over time
            if now is None:
                now = time.time()
            base = (now * 0.05) % 1.0
            pos = (bar_pos or 0.0)
            hue = (base + pos*0.7 + t*0.1) % 1.0
            s = 1.0
            v = 0.8 + 0.2*t
            r, g, b = self._hsv_to_rgb(hue, s, v)
            return self._ansi_color(r, g, b)
        else:
            # default to sunset stops
            stops = [
                (0.0, (75, 0, 110)),
                (0.45, (200, 0, 120)),
                (0.75, (255, 120, 0)),
                (1.0, (255, 220, 120)),
            ]
        # interpolate
        for i in range(1, len(stops)):
            t0, c0 = stops[i-1]
            t1, c1 = stops[i]
            if t <= t1:
                u = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                r = c0[0] + (c1[0]-c0[0]) * u
                g = c0[1] + (c1[1]-c0[1]) * u
                b = c0[2] + (c1[2]-c0[2]) * u
                return self._ansi_color(r, g, b)
        r, g, b = stops[-1][1]
        return self._ansi_color(r, g, b)

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        # h in [0,1], s,v in [0,1] => returns 0..255
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return int(r*255), int(g*255), int(b*255)

    def _render_frame(self, magnitudes):
        width, height = self._term_size()
        # Leave one line for instructions
        usable_height = max(4, height - 1)
        # Normalize magnitudes to 0..1
        x = np.clip(magnitudes * self.gain, 0, None)
        # Smooth rises and decays (simple leaky integrator per bar)
        higher = x > self.prev_vals
        self.prev_vals[higher] = self.prev_vals[higher] * self.rise + x[higher] * (1 - self.rise)
        self.prev_vals[~higher] = self.prev_vals[~higher] * self.decay + x[~higher] * (1 - self.decay)

        # Peak hold with slow decay
        self.peak_vals = np.maximum(self.peak_vals * self.peak_decay, self.prev_vals)

        # Decide how many bars to draw (never exceed terminal width)
        draw_bars = int(min(self.bars, max(1, width)))

        # Map to rows
        vals_rows_full = np.clip((self.prev_vals * usable_height).astype(int), 0, usable_height)
        peaks_rows_full = np.clip((self.peak_vals * usable_height).astype(int), 0, usable_height - 1)
        vals_rows = vals_rows_full[:draw_bars]
        peaks_rows = peaks_rows_full[:draw_bars]

        # Distribute terminal columns evenly across draw_bars to fill entire width
        base = width // draw_bars if draw_bars > 0 else 1
        rem = width % draw_bars if draw_bars > 0 else 0
        bar_widths = [base + 1 if i < rem else base for i in range(draw_bars)]

        # Peak style
        peak_char = '•' if self.cap == 'dot' else ('─' if self.cap == 'line' else ' ')
        COL_PEAK = '\x1b[97m'

        # Build frame top-down
        out_lines = []
        now = time.time()
        for row in range(usable_height, 0, -1):
            line_chars = []
            for i in range(draw_bars):
                w = bar_widths[i]
                filled = vals_rows[i] >= row
                peak_here = peaks_rows[i] == row
                if filled:
                    # Gradient color by vertical position and bar position
                    t = (row-1) / max(1, usable_height-1)
                    col = self._gradient_color(t, bar_pos=(i / max(1, draw_bars-1)), now=now)
                    seg = col + ('█' * w)
                elif peak_here and self.cap != 'none':
                    seg = COL_PEAK + (peak_char * max(1, min(1, w))) + (' ' * (w-1))
                else:
                    seg = ' ' * w
                line_chars.append(seg)
            out_lines.append(''.join(line_chars) + '\x1b[0m')

        # Footer/help
        out_lines.append("\x1b[0mPress 'q' or Ctrl+C to quit. ")
        sys.stdout.write('\x1b[H' + '\n'.join(out_lines))
        sys.stdout.flush()

    def run(self, audio: 'AudioLoopback', n_fft=4096):
        self._clear_and_home()
        try:
            # Precompute band mapping
            idxs = log_space_frequencies(self.bars, self.fmin, self.fmax, audio.samplerate, n_fft)
            window = np.hanning(n_fft).astype(np.float32)

            target_period = 1.0 / max(1, self.fps)
            next_time = time.perf_counter()

            # Double buffer for samples
            buf = np.zeros((0, audio.channels), dtype=np.float32)

            while True:
                chunk = audio.read(timeout=0.1)
                if chunk is not None:
                    buf = np.concatenate([buf, chunk], axis=0)

                # Process if enough samples
                if buf.shape[0] >= n_fft:
                    frame = buf[:n_fft, :]
                    buf = buf[n_fft:, :]
                    # Mixdown to mono
                    mono = frame.mean(axis=1)
                    # Apply window
                    mono = mono * window
                    # FFT magnitude
                    spec = np.fft.rfft(mono, n=n_fft)
                    mag = np.abs(spec)
                    # Convert to quasi-psd (square), then log-like scaling
                    psd = mag
                    # Aggregate into bars by taking max within bin neighborhoods
                    # For stability, average bins between midpoints
                    bands = np.zeros(self.bars, dtype=np.float32)
                    last = 0
                    for i, idx in enumerate(idxs):
                        a = last
                        b = idx
                        if b <= a:
                            b = a + 1
                        bands[i] = np.sqrt(np.mean(psd[a:b] ** 2) + 1e-12)
                        last = idx
                    # Normalize by a soft reference
                    bands = np.log1p(bands) / 6.0  # heuristic scaling
                    bands = np.clip(bands, 0.0, 1.0)

                    # Render at target fps
                    now = time.perf_counter()
                    if now >= next_time:
                        self._render_frame(bands)
                        next_time = now + target_period

                # Non-blocking keypress to quit
                if _kbhit():
                    ch = _getch()
                    if ch in ('q', 'Q'):
                        break
        finally:
            self._restore_cursor()


# Minimal non-blocking keyboard input for Windows
def _kbhit():
    if os.name != 'nt':
        return False
    try:
        import msvcrt
        return msvcrt.kbhit()
    except Exception:
        return False


def _getch():
    if os.name != 'nt':
        return ''
    try:
        import msvcrt
        ch = msvcrt.getch()
        try:
            return ch.decode('utf-8', 'ignore')
        except Exception:
            return ''
    except Exception:
        return ''


def parse_args():
    p = argparse.ArgumentParser(description="Minimal Windows CAVA-like visualizer (WASAPI loopback)")
    p.add_argument('--samplerate', type=int, default=48000, help='Sample rate (Hz)')
    p.add_argument('--blocksize', type=int, default=2048, help='Audio callback block size')
    p.add_argument('--channels', type=int, default=None, help='Force channel count (1 or 2 typically)')
    p.add_argument('--fft', type=int, default=4096, help='FFT size (power of two recommended)')
    p.add_argument('--bars', type=int, default=64, help='Number of bars')
    p.add_argument('--fps', type=int, default=60, help='Frames per second')
    p.add_argument('--height', type=int, default=None, help='Max terminal height to use')
    p.add_argument('--fmin', type=int, default=43, help='Minimum frequency (Hz)')
    p.add_argument('--fmax', type=int, default=16000, help='Maximum frequency (Hz)')
    p.add_argument('--decay', type=float, default=0.6, help='Decay smoothing 0..1 (higher = slower)')
    p.add_argument('--rise', type=float, default=0.5, help='Rise smoothing 0..1 (higher = slower)')
    p.add_argument('--gain', type=float, default=1.0, help='Additional gain factor')
    p.add_argument('--device', type=str, default=None, help='WASAPI output device index or name substring')
    p.add_argument('--list-devices', action='store_true', help='List devices and exit')
    p.add_argument('--debug', action='store_true', help='Print debug info')
    p.add_argument('--backend', choices=['auto', 'sd', 'sc'], default='auto', help='Audio backend: sounddevice (sd), soundcard (sc), or auto')
    p.add_argument('--theme', choices=['sunset','ocean','rainbow','mono'], default='sunset', help='Color theme')
    p.add_argument('--cap', choices=['dot','line','none'], default='dot', help='Peak indicator style')
    p.add_argument('--no-truecolor', action='store_true', help='Disable 24-bit colors (use 256-color)')
    return p.parse_args()


def main():
    _enable_vt_mode()
    args = parse_args()

    if args.list_devices:
        print("Host APIs:")
        for i, api in enumerate(sd.query_hostapis()):
            print(f"  [{i}] {api['name']}")
        print("\nDevices:")
        for i, d in enumerate(sd.query_devices()):
            print(f"  [{i}] {d['name']} (hostapi={d['hostapi']}, in={d['max_input_channels']}, out={d['max_output_channels']}, sr={int(d.get('default_samplerate') or 0)})")
        return

    # Parse device argument
    dev_index = None
    name_hint = None
    if args.device is not None:
        if args.device.isdigit():
            # Map numeric index to WASAPI variant if needed
            orig_index = int(args.device)
            dev_index = map_index_to_wasapi_output(orig_index)
            if args.debug and orig_index != dev_index:
                try:
                    dn = sd.query_devices(orig_index).get('name', '?')
                    wn = sd.query_devices(dev_index).get('name', '?')
                    print(f"[audio] mapped device index {orig_index} ('{dn}') -> WASAPI {dev_index} ('{wn}')")
                except Exception:
                    pass
        else:
            name_hint = args.device

    req_ch = args.channels if args.channels in (1, 2, 6, 8) else 2
    audio = AudioLoopback(
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        channels=req_ch,
        device=dev_index,
        name_hint=name_hint,
        debug=args.debug,
        backend=args.backend,
    )
    audio.start()

    vis = Visualizer(
        bars=args.bars,
        fps=args.fps,
        height=args.height,
        fmin=args.fmin,
        fmax=args.fmax,
        decay=args.decay,
        rise=args.rise,
        gain=args.gain,
        theme=args.theme,
        cap=args.cap,
        truecolor=not args.no_truecolor,
    )

    try:
        vis.run(audio=audio, n_fft=args.fft)
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()


if __name__ == '__main__':
    main()
