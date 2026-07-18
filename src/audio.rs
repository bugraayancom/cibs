use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::{Error, Result};

/// Target sample rate expected by the mel frontend.
pub const SAMPLE_RATE: u32 = 16_000;

/// Decode any audio file to 16 kHz mono f32 PCM.
///
/// Uses the system `ffmpeg` binary (handles wav/mp3/m4a/flac/ogg/...).
/// Falls back to a built-in WAV reader with linear resampling when
/// `ffmpeg` is not installed.
pub fn decode_file(path: &Path) -> Result<Vec<f32>> {
    if !path.is_file() {
        return Err(Error::Audio(format!("file not found: {}", path.display())));
    }
    match decode_with_ffmpeg(path) {
        Ok(samples) => Ok(samples),
        Err(ffmpeg_err) => {
            if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
            {
                decode_wav(path)
            } else {
                Err(ffmpeg_err)
            }
        }
    }
}

/// Decode in-memory audio bytes (e.g. an HTTP upload). The original file
/// name is used as an extension hint for the demuxer.
pub fn decode_bytes(bytes: &[u8], filename_hint: &str) -> Result<Vec<f32>> {
    let ext = Path::new(filename_hint)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    let tmp = std::env::temp_dir().join(format!(
        "qwen3-asr-{}-{}.{ext}",
        std::process::id(),
        unique_id()
    ));
    std::fs::write(&tmp, bytes).map_err(|e| Error::io(&tmp, e))?;
    let result = decode_file(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result
}

fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn decode_with_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Audio(format!("failed to spawn ffmpeg: {e}")))?;

    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout piped")
        .read_to_end(&mut raw)
        .map_err(|e| Error::Audio(format!("failed to read ffmpeg output: {e}")))?;

    let mut stderr = String::new();
    if let Some(mut err_pipe) = child.stderr.take() {
        let _ = err_pipe.read_to_string(&mut stderr);
    }
    let status = child
        .wait()
        .map_err(|e| Error::Audio(format!("ffmpeg wait failed: {e}")))?;
    if !status.success() {
        return Err(Error::Audio(format!(
            "ffmpeg failed on {}: {}",
            path.display(),
            stderr.trim()
        )));
    }

    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn decode_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| Error::Audio(format!("wav open: {e}")))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Audio(format!("wav read: {e}")))?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error::Audio(format!("wav read: {e}")))?
        }
    };

    // Downmix to mono.
    let channels = spec.channels as usize;
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    Ok(resample_linear(&mono, spec.sample_rate, SAMPLE_RATE))
}

/// Simple linear resampler (fallback path only; ffmpeg handles the rest).
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if input.is_empty() || from == to {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = input[idx];
            let b = input[(idx + 1).min(input.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_constant_signal() {
        let input = vec![0.5f32; 48_000];
        let out = resample_linear(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
}
