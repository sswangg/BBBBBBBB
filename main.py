import os
import subprocess

import numpy as np

from config import LED_COUNT, SAMPLING_FREQ
from visualizer import FFTRainbow

FRAME_MS = 100
AUDIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lace.mp3")
FRAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "led_frames.npy")


def load_mono_samples(path, sampling_freq=SAMPLING_FREQ):
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sampling_freq),
            "-",
        ],
        stderr=subprocess.DEVNULL,
    )
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


def generate_led_frames(samples, visualizer, frame_ms=FRAME_MS, sampling_freq=SAMPLING_FREQ):
    hop = max(1, int(round(sampling_freq * frame_ms / 1000.0)))
    window = visualizer.required_samples
    padded = np.pad(samples, (window, 0), mode="constant")
    frames = []

    for end in range(window, len(padded) + 1, hop):
        sample_array = padded[end - window:end]
        color_array = visualizer.visualize(sample_array)
        frames.append(np.clip(color_array, 0, 255).astype(np.uint8))

    frames = np.stack(frames, axis=0)
    if frames.shape[1] != LED_COUNT:
        raise ValueError(f"Expected LED arrays of length {LED_COUNT}, got {frames.shape[1]}")
    return frames


if __name__ == "__main__":
    samples = load_mono_samples(AUDIO_PATH)
    visualizer = FFTRainbow()
    frames = generate_led_frames(samples, visualizer)[:10]
    np.set_printoptions(threshold=np.inf)
    with open(FRAMES_PATH, "w") as f:
        f.write(repr(frames))
    print(f"Wrote {frames.shape[0]} frames of shape ({LED_COUNT}, 3) to {FRAMES_PATH}")
    print(f"Each frame is {FRAME_MS} ms ({frames.shape[0] * FRAME_MS / 1000.0:.2f} s total)")
