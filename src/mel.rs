use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

use crate::error::{Error, Result};

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;

/// Log-mel spectrogram in Whisper layout: `data[mel * n_frames + frame]`.
pub struct LogMelSpectrogram {
    pub data: Vec<f32>,
    pub n_mels: usize,
    /// Total frames, zero-audio-padded to a multiple of the encoder chunk.
    pub n_frames: usize,
    /// Frames covering the actual (unpadded) audio.
    pub n_valid_frames: usize,
}

/// Compute a Whisper-compatible 128-bin log-mel spectrogram.
///
/// Matches `WhisperFeatureExtractor` (hann window, n_fft=400, hop=160,
/// center-reflect padding, power spectrum, slaney mel filterbank, log10,
/// dynamic-range clamp to max-8, then `(x+4)/4`).
///
/// The raw audio is zero-padded so the frame count is a multiple of
/// `pad_multiple_frames` (the encoder's conv chunk length); padded frames are
/// reported via `n_valid_frames` so they can be masked out later.
pub fn log_mel_spectrogram(
    samples: &[f32],
    n_mels: usize,
    pad_multiple_frames: usize,
) -> Result<LogMelSpectrogram> {
    if samples.len() < N_FFT / 2 + 1 {
        return Err(Error::Audio(format!(
            "audio too short: {} samples (need at least {})",
            samples.len(),
            N_FFT / 2 + 1
        )));
    }

    let n_valid_frames = samples.len().div_ceil(HOP_LENGTH);
    let pad_samples = pad_multiple_frames * HOP_LENGTH;
    let padded_len = samples.len().div_ceil(pad_samples) * pad_samples;
    let mut audio = samples.to_vec();
    audio.resize(padded_len, 0.0);
    let n_frames = padded_len / HOP_LENGTH;

    // torch.stft(center=True): reflect-pad by n_fft/2 on both sides.
    let half = N_FFT / 2;
    let mut padded = Vec::with_capacity(audio.len() + N_FFT);
    padded.extend((1..=half).rev().map(|i| audio[i]));
    padded.extend_from_slice(&audio);
    padded.extend(
        (audio.len() - half - 1..audio.len() - 1)
            .rev()
            .map(|i| audio[i]),
    );

    // Periodic hann window.
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| {
            let x = (std::f64::consts::PI * i as f64 / N_FFT as f64).sin();
            (x * x) as f32
        })
        .collect();

    let n_freqs = N_FFT / 2 + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut fft_in = fft.make_input_vec();
    let mut fft_out = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();

    // Power spectrogram: [frame][freq].
    let mut power = vec![0.0f32; n_frames * n_freqs];
    for t in 0..n_frames {
        let start = t * HOP_LENGTH;
        for i in 0..N_FFT {
            fft_in[i] = padded[start + i] * window[i];
        }
        fft.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
            .map_err(|e| Error::Audio(format!("fft failed: {e}")))?;
        let row = &mut power[t * n_freqs..(t + 1) * n_freqs];
        for (p, c) in row.iter_mut().zip(fft_out.iter()) {
            *p = c.norm_sqr();
        }
    }
    let _ = Complex32::default(); // keep realfft's complex type in scope

    // Mel projection + log10.
    let filters = mel_filterbank(n_mels, n_freqs, SAMPLE_RATE_F64);
    let mut log_spec = vec![0.0f32; n_mels * n_frames];
    let mut global_max = f32::MIN;
    for (m, filter) in filters.chunks_exact(n_freqs).enumerate() {
        // Filters are sparse (triangular); restrict to the nonzero band.
        let lo = filter.iter().position(|&v| v > 0.0).unwrap_or(0);
        let hi = filter.iter().rposition(|&v| v > 0.0).map_or(0, |i| i + 1);
        for t in 0..n_frames {
            let row = &power[t * n_freqs..(t + 1) * n_freqs];
            let mut acc = 0.0f32;
            for k in lo..hi {
                acc += filter[k] * row[k];
            }
            let v = acc.max(1e-10).log10();
            global_max = global_max.max(v);
            log_spec[m * n_frames + t] = v;
        }
    }

    let floor = global_max - 8.0;
    for v in log_spec.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }

    Ok(LogMelSpectrogram {
        data: log_spec,
        n_mels,
        n_frames,
        n_valid_frames,
    })
}

const SAMPLE_RATE_F64: f64 = 16_000.0;

/// Slaney-scale mel filterbank with slaney area normalization
/// (librosa `filters.mel(htk=False, norm="slaney")`).
/// Returns `n_mels * n_freqs` weights, row-major per mel bin.
fn mel_filterbank(n_mels: usize, n_freqs: usize, sample_rate: f64) -> Vec<f32> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep: f64 = (6.4f64).ln() / 27.0;

    let hz_to_mel = |hz: f64| -> f64 {
        if hz >= MIN_LOG_HZ {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
        } else {
            hz / F_SP
        }
    };
    let mel_to_hz = |mel: f64| -> f64 {
        if mel >= MIN_LOG_MEL {
            MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
        } else {
            mel * F_SP
        }
    };

    let f_max = sample_rate / 2.0;
    let mel_max = hz_to_mel(f_max);
    // n_mels + 2 boundary frequencies.
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_max * i as f64 / (n_mels + 1) as f64))
        .collect();

    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| f_max * i as f64 / (n_freqs - 1) as f64)
        .collect();

    let mut weights = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let (f_lo, f_mid, f_hi) = (mel_f[m], mel_f[m + 1], mel_f[m + 2]);
        let enorm = 2.0 / (f_hi - f_lo);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - f_lo) / (f_mid - f_lo);
            let upper = (f_hi - freq) / (f_hi - f_mid);
            let w = lower.min(upper).max(0.0);
            weights[m * n_freqs + k] = (w * enorm) as f32;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_rows_are_nonempty() {
        let fb = mel_filterbank(128, 201, 16_000.0);
        for m in 0..128 {
            let row = &fb[m * 201..(m + 1) * 201];
            assert!(row.iter().any(|&v| v > 0.0), "mel filter {m} is all zeros");
        }
    }

    #[test]
    fn frame_counts_and_padding() {
        // 1.5 s of a 440 Hz tone -> 150 valid frames, padded to 200.
        let samples: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&samples, 128, 100).unwrap();
        assert_eq!(mel.n_valid_frames, 150);
        assert_eq!(mel.n_frames, 200);
        assert_eq!(mel.data.len(), 128 * 200);
        // Values are finite; Whisper normalization keeps an 8-dB dynamic
        // range: max - min <= 8/4 = 2 after (x+4)/4 scaling.
        assert!(mel.data.iter().all(|v| v.is_finite()));
        let max = mel.data.iter().cloned().fold(f32::MIN, f32::max);
        let min = mel.data.iter().cloned().fold(f32::MAX, f32::min);
        assert!((max - min) <= 2.0 + 1e-5);
    }
}
