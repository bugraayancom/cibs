# Generates reference log-mel output following HuggingFace
# WhisperFeatureExtractor's numpy implementation exactly.
# Run: python3 gen_mel_reference.py  (needs numpy only)
import numpy as np

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 128


def hertz_to_mel(freq):
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    log_region = freq >= min_log_hertz
    mels = np.where(
        log_region,
        min_log_mel + np.log(np.maximum(freq, min_log_hertz) / min_log_hertz) * logstep,
        mels,
    )
    return mels


def mel_to_hertz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freq = np.where(
        log_region, min_log_hertz * np.exp(logstep * (mels - min_log_mel)), freq
    )
    return freq


def mel_filter_bank():
    fft_freqs = np.linspace(0, SR // 2, 1 + N_FFT // 2)
    mel_min = hertz_to_mel(0.0)
    mel_max = hertz_to_mel(float(SR // 2))
    mel_freqs = np.linspace(mel_min, mel_max, N_MELS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (filter_freqs[2 : N_MELS + 2] - filter_freqs[:N_MELS])
    fb *= np.expand_dims(enorm, 0)
    return fb  # (n_freqs, n_mels)


def log_mel(waveform):
    window = np.hanning(N_FFT + 1)[:-1]  # periodic hann
    pad = N_FFT // 2
    wf = np.pad(waveform, pad, mode="reflect")
    num_frames = 1 + (len(wf) - N_FFT) // HOP
    frames = np.stack(
        [wf[i * HOP : i * HOP + N_FFT] * window for i in range(num_frames)]
    )
    stft = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = np.abs(stft) ** 2  # (frames, freqs)
    mel = power @ mel_filter_bank()  # (frames, mels)
    log_spec = np.log10(np.maximum(mel, 1e-10)).T  # (mels, frames)
    log_spec = log_spec[:, :-1]  # whisper drops the last frame
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def lcg_noise(n, seed=42):
    # Deterministic LCG so the Rust test can regenerate the same input.
    state = np.uint64(seed)
    out = np.empty(n, dtype=np.float32)
    a = np.uint64(6364136223846793005)
    c = np.uint64(1442695040888963407)
    for i in range(n):
        state = state * a + c  # wraps mod 2**64
        out[i] = (np.float64(state >> np.uint64(40)) / np.float64(1 << 24)) * 2.0 - 1.0
    return out


def main():
    # Multiple of one encoder chunk (16000 samples) so the Rust pipeline adds
    # no zero padding and every frame is comparable at tight tolerance.
    n = 2 * SR
    t = np.arange(n, dtype=np.float64) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 1333.0 * t)
    audio = (tone.astype(np.float32) + 0.05 * lcg_noise(n)).astype(np.float32)

    mel = log_mel(audio)
    audio.tofile("mel_ref_input.f32")
    mel.tofile("mel_ref_expected.f32")
    print("input samples:", n, "mel shape:", mel.shape)


if __name__ == "__main__":
    main()
