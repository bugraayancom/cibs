use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use clap::Parser;
use tracing_subscriber::EnvFilter;

use qwen3_asr::asr::AsrModel;
use qwen3_asr::audio::SAMPLE_RATE;
use qwen3_asr::translate::Translator;

#[derive(Debug, Parser)]
#[command(
    name = "cibs",
    about = "CİBS (CİB Simultane) — yerel transkripsiyon ve anlık çeviri CLI (MLX / Apple Silicon)"
)]
struct Args {
    /// Path to a HuggingFace model directory (config.json + safetensors + vocab).
    model_dir: PathBuf,

    /// Audio file to transcribe (wav/mp3/m4a/flac/...). Omit with --live.
    audio: Option<PathBuf>,

    /// Live mode: capture the microphone and transcribe continuously.
    #[arg(long)]
    live: bool,

    /// avfoundation audio device index for --live (see
    /// `ffmpeg -f avfoundation -list_devices true -i ""`).
    #[arg(long, default_value = "0")]
    device: String,

    /// Seconds of new audio per streaming step in --live mode.
    #[arg(long, default_value_t = 2.0)]
    chunk_secs: f32,

    /// Segment length in --live mode; after this much audio the text is
    /// committed and the buffer restarts (bounds re-encoding cost).
    #[arg(long, default_value_t = 25.0)]
    segment_secs: f32,

    /// Force language (`en`, `tr`, `Chinese`, ...). Omit for auto-detect.
    #[arg(long, short)]
    language: Option<String>,

    /// Translate the transcript into this language (`tr`, `en`, `German`, ...).
    /// Requires --translator-dir.
    #[arg(long)]
    translate_to: Option<String>,

    /// Qwen3 instruct model directory used for local translation
    /// (e.g. a downloaded Qwen3-0.6B).
    #[arg(long)]
    translator_dir: Option<PathBuf>,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// Print language / RTF / timing to stderr.
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let result = if args.live {
        run_live(args)
    } else {
        run_file(args)
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn load_translator(args: &Args) -> anyhow::Result<Option<Translator>> {
    match (&args.translate_to, &args.translator_dir) {
        (Some(_), Some(dir)) => Ok(Some(Translator::load(dir)?)),
        (Some(_), None) => anyhow::bail!("--translate-to requires --translator-dir"),
        _ => Ok(None),
    }
}

fn resolve_target(target: &str) -> String {
    qwen3_asr::config::resolve_language(Some(target))
        .ok()
        .flatten()
        .unwrap_or_else(|| target.to_string())
}

fn run_file(args: Args) -> anyhow::Result<()> {
    let audio = args
        .audio
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("audio file required (or use --live)"))?;
    let mut translator = load_translator(&args)?;
    let mut model = AsrModel::load(&args.model_dir)?;
    let result = model.transcribe_file(audio, args.language.as_deref(), args.max_tokens)?;

    if args.verbose {
        eprintln!(
            "language={:?} audio={:.2}s elapsed={:.2}s rtf={:.3} audio_tokens={} gen_tokens={}",
            result.language,
            result.audio_seconds,
            result.elapsed_seconds,
            result.rtf,
            result.num_audio_tokens,
            result.num_generated_tokens
        );
    }

    println!("{}", result.text);

    if let (Some(target), Some(tr)) = (&args.translate_to, translator.as_mut()) {
        let translated = tr.translate(&result.text, &resolve_target(target))?;
        println!("--- {target} ---");
        println!("{translated}");
    }
    Ok(())
}

/// Streaming parameters from the official Qwen3-ASR recipe: the first
/// N chunks are decoded from scratch; afterwards the previous response is
/// kept as a forced prefix minus the last few (unstable) tokens.
const UNFIXED_CHUNK_NUM: usize = 2;
const UNFIXED_TOKEN_NUM: usize = 5;

fn run_live(args: Args) -> anyhow::Result<()> {
    let mut translator = load_translator(&args)?;
    let target_name = args.translate_to.as_deref().map(resolve_target);
    let mut model = AsrModel::load(&args.model_dir)?;
    eprintln!("model hazır — mikrofon dinleniyor (bitirmek için Ctrl+C)...");

    let rx = spawn_mic_reader(&args.device)?;

    let chunk_samples = (args.chunk_secs.max(0.5) * SAMPLE_RATE as f32) as usize;
    let segment_samples = (args.segment_secs.max(5.0) * SAMPLE_RATE as f32) as usize;
    let min_samples = SAMPLE_RATE as usize / 2; // 0.5 s before the first step

    let mut buffer: Vec<f32> = Vec::new();
    let mut prev_tokens: Vec<u32> = Vec::new();
    let mut step_count = 0usize;
    let mut last_step_len = 0usize;
    let mut committed = String::new();
    let stdout = std::io::stdout();

    loop {
        // Block for the next audio block, then drain whatever else arrived.
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(block) => buffer.extend_from_slice(&block),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                anyhow::bail!("mikrofon akışı kapandı (ffmpeg sonlandı)");
            }
        }
        while let Ok(block) = rx.try_recv() {
            buffer.extend_from_slice(&block);
        }

        if buffer.len() < min_samples || buffer.len() < last_step_len + chunk_samples {
            continue;
        }

        let t0 = Instant::now();
        let forced: Vec<u32> = if step_count < UNFIXED_CHUNK_NUM {
            Vec::new()
        } else {
            let keep = prev_tokens.len().saturating_sub(UNFIXED_TOKEN_NUM);
            prev_tokens[..keep].to_vec()
        };

        let (tokens, tr) =
            model.transcribe_stream_step(&buffer, args.language.as_deref(), &forced, 256)?;
        prev_tokens = tokens;
        last_step_len = buffer.len();
        step_count += 1;

        // Rewrite the current (unstable) line in place.
        let line = tr.text.replace('\n', " ");
        let mut out = stdout.lock();
        write!(out, "\r\x1b[2K{line}")?;
        out.flush()?;

        if args.verbose {
            eprintln!(
                " [adım {step_count}: {:.1}s ses, {:.2}s işlem, rtf {:.3}]",
                tr.audio_seconds,
                t0.elapsed().as_secs_f64(),
                tr.rtf
            );
        }

        // Segment full: commit the text and restart the audio buffer.
        if buffer.len() >= segment_samples {
            {
                let mut out = stdout.lock();
                writeln!(out, "\r\x1b[2K{line}")?;
                out.flush()?;
            }
            if let (Some(tr), Some(target)) = (translator.as_mut(), target_name.as_deref()) {
                let translated = tr.translate(&line, target)?;
                let mut out = stdout.lock();
                writeln!(out, "  → {translated}")?;
                out.flush()?;
            }
            if !committed.is_empty() {
                committed.push(' ');
            }
            committed.push_str(&line);
            buffer.clear();
            prev_tokens.clear();
            step_count = 0;
            last_step_len = 0;
        }
    }
}

/// Spawn ffmpeg capturing the default/selected mic as 16 kHz mono f32le and
/// stream ~100 ms blocks over a channel.
fn spawn_mic_reader(device: &str) -> anyhow::Result<mpsc::Receiver<Vec<f32>>> {
    let mut child = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-f", "avfoundation", "-i"])
        .arg(format!(":{device}"))
        .args([
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-f",
            "f32le",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .stdin(std::process::Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("ffmpeg başlatılamadı: {e}"))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout alınamadı"))?;

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    std::thread::spawn(move || {
        // 100 ms of f32 samples per block.
        let block_bytes = (SAMPLE_RATE as usize / 10) * 4;
        let mut buf = vec![0u8; block_bytes];
        loop {
            let mut filled = 0;
            while filled < block_bytes {
                match stdout.read(&mut buf[filled..]) {
                    Ok(0) => {
                        let _ = child.wait();
                        return;
                    }
                    Ok(n) => filled += n,
                    Err(_) => {
                        let _ = child.wait();
                        return;
                    }
                }
            }
            let samples: Vec<f32> = buf
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if tx.send(samples).is_err() {
                let _ = child.kill();
                return;
            }
        }
    });

    Ok(rx)
}
