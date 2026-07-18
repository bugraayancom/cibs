use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use qwen3_asr::asr::AsrModel;
use qwen3_asr::audio::SAMPLE_RATE;
use qwen3_asr::translate::Translator;

#[derive(Debug, Parser)]
#[command(
    name = "cibs-server",
    about = "CİBS (CİB Simultane) — yerel canlı çeviri sunucusu (OpenAI uyumlu API)"
)]
struct Args {
    /// Path to a HuggingFace model directory.
    #[arg(long)]
    model_dir: PathBuf,

    /// Optional Qwen3 instruct model directory used for local translation
    /// (e.g. Qwen3-0.6B). When set, requests may pass `translate_to`.
    #[arg(long)]
    translator_dir: Option<PathBuf>,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Optional Bearer API key. When set, requests must send
    /// `Authorization: Bearer <key>`.
    #[arg(long)]
    api_key: Option<String>,

    /// Maximum new tokens per request.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// Public model id advertised by `/v1/models`.
    #[arg(long, default_value = "cibs")]
    model_id: String,
}

/// Both models live behind a single lock: MLX does not tolerate concurrent
/// GPU evaluation from multiple threads (segfaults), so all inference —
/// transcription steps and translations alike — must be serialized.
struct Engines {
    asr: AsrModel,
    translator: Option<Translator>,
}

struct AppState {
    engines: Mutex<Engines>,
    has_translator: bool,
    /// Last `(text, target, translation)`; live mode often re-sends the same
    /// transcript (e.g. during pauses), so this skips redundant generation.
    translation_cache: Mutex<Option<(String, String, String)>>,
    api_key: Option<String>,
    max_tokens: usize,
    model_id: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    info!(path = %args.model_dir.display(), "loading model");
    let mut model = AsrModel::load(&args.model_dir)?;
    info!("model ready");

    let mut translator = match &args.translator_dir {
        Some(dir) => Some(Translator::load(dir)?),
        None => None,
    };

    // Warmup: the first request otherwise pays several seconds of Metal
    // kernel compilation. Run a dummy pass through both models now.
    {
        let t0 = Instant::now();
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        if let Err(e) = model.transcribe_samples(&silence, None, 4) {
            warn!(error = %e, "ASR warmup failed");
        }
        if let Some(tr) = translator.as_mut() {
            if let Err(e) = tr.translate("Hello.", "Turkish") {
                warn!(error = %e, "translator warmup failed");
            }
        }
        info!(
            elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
            "warmup done"
        );
    }

    let state = Arc::new(AppState {
        has_translator: translator.is_some(),
        engines: Mutex::new(Engines {
            asr: model,
            translator,
        }),
        translation_cache: Mutex::new(None),
        api_key: args.api_key,
        max_tokens: args.max_tokens,
        model_id: args.model_id,
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/logo.png", get(logo))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/translate", post(translate_text))
        .route("/v1/live", get(live_ws))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!(%addr, "listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Minimal browser test page served at `/`.
async fn index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
}

/// İletişim Başkanlığı logo, embedded in the binary.
async fn logo() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        LOGO_PNG,
    )
}

const LOGO_PNG: &[u8] = include_bytes!("../../assets/logo.png");

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CİBS — CİB Simultane</title>
<link rel="icon" type="image/png" href="/logo.png">
<style>
  :root {
    color-scheme: light dark;
    --red: #E30A17;
    --red-dark: #B00812;
    --gray: #8C8C8C;
  }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 760px;
         margin: 1.6rem auto; padding: 0 1rem; line-height: 1.5; }
  header { display: flex; align-items: center; gap: 1rem; margin-bottom: .4rem; }
  header img { width: 72px; height: 72px; }
  .brand h1 { font-size: 1.7rem; margin: 0; letter-spacing: .04em; }
  .brand h1 .red { color: var(--red); }
  .brand p { margin: .1rem 0 0; color: var(--gray); font-size: .95rem; }
  .privacy { border-left: 3px solid var(--red); padding: .4rem .8rem;
             color: var(--gray); font-size: .85rem; margin: 1rem 0; }
  .row { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center;
         margin: 1rem 0; }
  select { padding: .5rem .9rem; border-radius: 10px; border: 1px solid #8884;
           background: transparent; font-size: 1rem; }
  button { padding: .6rem 1.5rem; border-radius: 10px; border: none;
           background: var(--red); color: #fff; font-size: 1.05rem;
           cursor: pointer; font-weight: 700; }
  button:hover { background: var(--red-dark); }
  button.rec { background: transparent; color: var(--red);
               border: 2px solid var(--red); }
  .label { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
           color: var(--gray); margin: 1.1rem 0 .3rem; font-weight: 600; }
  .out { white-space: pre-wrap; border: 1px solid #8884; border-radius: 10px;
         padding: 1rem; min-height: 3.2rem; }
  #trOut { font-size: 1.25rem; min-height: 5rem; border: 2px solid var(--red);
           border-radius: 12px; }
  .muted { opacity: .6; }
  #stat { font-size: .85rem; color: var(--gray); }
  #dot { display: inline-block; width: .65rem; height: .65rem; border-radius: 50%;
         background: var(--red); margin-right: .45rem; visibility: hidden;
         animation: blink 1s infinite; vertical-align: baseline; }
  @keyframes blink { 50% { opacity: .25; } }
  footer { margin-top: 2rem; color: var(--gray); font-size: .8rem;
           border-top: 1px solid #8883; padding-top: .8rem; }
</style>
</head>
<body>
<header>
  <img src="/logo.png" alt="İletişim Başkanlığı">
  <div class="brand">
    <h1><span id="dot"></span><span class="red">CİBS</span></h1>
    <p>CİB Simultane — Yerel Anlık Çeviri</p>
  </div>
</header>
<div class="privacy">Tüm işlem bu cihazda gerçekleşir; hiçbir ses veya metin
buluta gönderilmez.</div>

<div class="row">
  <button id="mic">Dinlemeye başla</button>
  <select id="target">
    <option value="tr" selected>Türkçeye çevir</option>
    <option value="en">İngilizceye çevir</option>
    <option value="de">Almancaya çevir</option>
    <option value="fr">Fransızcaya çevir</option>
    <option value="es">İspanyolcaya çevir</option>
    <option value="zh">Çinceye çevir</option>
    <option value="ru">Rusçaya çevir</option>
    <option value="ar">Arapçaya çevir</option>
    <option value="">Çeviri yok (sadece yazıya dök)</option>
  </select>
  <select id="srcLang">
    <option value="">Konuşma dili: otomatik</option>
    <option value="tr">Türkçe</option>
    <option value="en">İngilizce</option>
    <option value="zh">Çince</option>
    <option value="de">Almanca</option>
    <option value="fr">Fransızca</option>
    <option value="es">İspanyolca</option>
    <option value="ru">Rusça</option>
    <option value="ar">Arapça</option>
    <option value="ja">Japonca</option>
    <option value="ko">Korece</option>
  </select>
  <span id="stat"></span>
</div>

<div class="label">Çeviri</div>
<div id="trOut" class="out muted">Dinlemeye başlayınca çeviri burada görünecek.</div>

<div class="label">Duyulan (transkript)</div>
<div id="asrOut" class="out muted">…</div>

<footer>CİBS &mdash; CİB Simultane &middot; T.C. Cumhurbaşkanlığı İletişim
Başkanlığı &middot; Tamamen çevrimdışı çalışır.</footer>

<script>
// Live pipeline: an AudioWorklet captures raw microphone PCM, ~250 ms slices
// are streamed over a WebSocket to /v1/live, and the server runs a stateful
// forced-prefix decode so each step only generates the newest tokens.
// Translation runs in a separate loop against /v1/translate so it never
// blocks transcription.
// Translation shares the GPU with transcription (single lock), so partial
// translations run sparsely to avoid starving the live transcript.
const TR_MS = 2500;
const micBtn = document.getElementById('mic');
const asrOut = document.getElementById('asrOut');
const trOut = document.getElementById('trOut');
const stat = document.getElementById('stat');
const dot = document.getElementById('dot');
let ws = null, audioCtx = null, worklet = null, mediaStream = null;
let running = false, trTimer = null, trBusy = false, lastTranslated = null;
let segTexts = [], segTrs = [];   // committed (frozen) segments
let curText = '', curTr = '';     // current segment, still changing

function render() {
  const t = segTexts.concat(curText ? [curText] : []).join(' ');
  const tr = segTrs.concat(curTr ? [curTr] : []).join(' ');
  asrOut.classList.toggle('muted', !t);
  asrOut.textContent = t || '…';
  const target = document.getElementById('target').value;
  if (target) {
    trOut.classList.toggle('muted', !tr);
    trOut.textContent = tr || '…';
  } else {
    trOut.classList.add('muted');
    trOut.textContent = '(çeviri kapalı)';
  }
}

async function fetchTranslation(text, target) {
  const r = await fetch('/v1/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, target }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error?.message || r.status);
  return j.translation;
}

// Continuously translates the current partial transcript, independent of the
// transcription stream.
async function translateStep() {
  const target = document.getElementById('target').value;
  if (!target || trBusy) return;
  const text = curText;
  if (!text || text === lastTranslated) return;
  trBusy = true;
  try {
    curTr = await fetchTranslation(text, target);
    lastTranslated = text;
    render();
  } catch (err) {
    // transient; next tick retries
  } finally {
    trBusy = false;
  }
}

// Re-translate a frozen segment with its final text (replaces the possibly
// stale partial translation).
function finalizeSegment(idx, text) {
  const target = document.getElementById('target').value;
  if (!target || !text) return;
  fetchTranslation(text, target)
    .then((tr) => { segTrs[idx] = tr; render(); })
    .catch(() => {});
}

// --- microphone capture: AudioWorklet → 16 kHz f32 PCM → WebSocket ---
const workletCode = `
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
`;

function resampleTo16k(f32, srcRate) {
  if (srcRate === 16000) return f32;
  const ratio = srcRate / 16000;
  const n = Math.floor(f32.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const frac = x - i0;
    const a = f32[i0];
    const b = f32[Math.min(i0 + 1, f32.length - 1)];
    out[i] = a + (b - a) * frac;
  }
  return out;
}

let pcmParts = [], pcmLen = 0;
function onPcm(chunk) {
  pcmParts.push(chunk);
  pcmLen += chunk.length;
  if (pcmLen < audioCtx.sampleRate / 4) return; // ~250 ms slices
  const all = new Float32Array(pcmLen);
  let o = 0;
  for (const p of pcmParts) { all.set(p, o); o += p.length; }
  pcmParts = []; pcmLen = 0;
  const out = resampleTo16k(all, audioCtx.sampleRate);
  if (ws && ws.readyState === 1) ws.send(out.buffer);
}

function handleServerMsg(e) {
  let j;
  try { j = JSON.parse(e.data); } catch (_) { return; }
  if (j.type === 'partial') {
    curText = j.text || '';
    render();
    stat.textContent = (j.audio_s?.toFixed(0) || '?') + ' sn ses · adım ' +
                       ((j.step_ms || 0) / 1000).toFixed(1) + ' sn' +
                       (j.language ? ' · ' + j.language : '');
  } else if (j.type === 'commit') {
    const idx = segTexts.length;
    segTexts.push(j.text || '');
    segTrs.push(curTr || '…');
    finalizeSegment(idx, j.text || '');
    curText = ''; curTr = ''; lastTranslated = null;
    render();
  } else if (j.type === 'error') {
    stat.textContent = 'hata: ' + j.message;
  }
}

async function startLive() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
  } catch (err) {
    trOut.textContent = 'Mikrofon izni alınamadı: ' + err.message;
    return false;
  }
  audioCtx = new AudioContext({ sampleRate: 16000 });
  const url = URL.createObjectURL(new Blob([workletCode], { type: 'application/javascript' }));
  await audioCtx.audioWorklet.addModule(url);
  const src = audioCtx.createMediaStreamSource(mediaStream);
  worklet = new AudioWorkletNode(audioCtx, 'pcm-capture');
  worklet.port.onmessage = (e) => onPcm(e.data);
  src.connect(worklet);

  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  ws = new WebSocket(proto + location.host + '/v1/live');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => ws.send(JSON.stringify({
    type: 'config',
    language: document.getElementById('srcLang').value || null,
  }));
  ws.onmessage = handleServerMsg;
  ws.onclose = () => { if (running) stopLive(); };
  return true;
}

function stopLive() {
  running = false;
  clearInterval(trTimer); trTimer = null;
  if (ws && ws.readyState === 1) {
    try { ws.send(JSON.stringify({ type: 'stop' })); } catch (_) {}
  }
  const w = ws; ws = null;
  // Give the server a moment to send the final partial + commit.
  if (w) setTimeout(() => { try { w.close(); } catch (_) {} }, 2500);
  if (worklet) { worklet.disconnect(); worklet = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  pcmParts = []; pcmLen = 0;
  micBtn.textContent = 'Dinlemeye başla';
  micBtn.classList.remove('rec');
  dot.style.visibility = 'hidden';
  stat.textContent += ' · bitti';
}

micBtn.addEventListener('click', async () => {
  if (running) { stopLive(); return; }
  segTexts = []; segTrs = []; curText = ''; curTr = ''; lastTranslated = null;
  if (!await startLive()) return;
  running = true;
  trTimer = setInterval(translateStep, TR_MS);
  micBtn.textContent = 'Durdur';
  micBtn.classList.add('rec');
  dot.style.visibility = 'visible';
  asrOut.textContent = 'Dinleniyor…';
  asrOut.classList.add('muted');
  trOut.textContent = '…';
  trOut.classList.add('muted');
  stat.textContent = '';
});
</script>
</body>
</html>"#;

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id.clone(),
            object: "model",
            owned_by: "local",
        }],
    })
}

#[derive(Serialize)]
struct TranscriptionJson {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    translation: Option<String>,
}

#[derive(Serialize)]
struct VerboseTranscription {
    text: String,
    language: Option<String>,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    translation: Option<String>,
}

async fn transcribe(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&headers, state.api_key.as_deref())?;

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = String::from("audio.wav");
    let mut language: Option<String> = None;
    let mut response_format = String::from("json");
    let mut translate_to: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                if let Some(name) = field.file_name() {
                    filename = name.to_string();
                }
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("file read: {e}")))?;
                file_bytes = Some(data.to_vec());
            }
            "language" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("language: {e}")))?;
                if !v.is_empty() {
                    language = Some(v);
                }
            }
            "translate_to" => {
                let v = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("translate_to: {e}")))?;
                if !v.is_empty() {
                    translate_to = Some(v);
                }
            }
            "response_format" => {
                response_format = field
                    .text()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("response_format: {e}")))?;
            }
            "model" => {
                let _ = field.bytes().await;
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let bytes = file_bytes.ok_or_else(|| ApiError::bad_request("missing form field `file`"))?;

    if translate_to.is_some() && !state.has_translator {
        return Err(ApiError::bad_request(
            "translation not available: start the server with --translator-dir",
        ));
    }

    let max_tokens = state.max_tokens;
    let mut engines = state.engines.lock().await;
    let result = engines
        .asr
        .transcribe_bytes(&bytes, &filename, language.as_deref(), max_tokens)
        .map_err(|e| ApiError::internal(e.to_string()))?;

    // Optional local translation of the transcript.
    let translation = match (&translate_to, engines.translator.as_mut()) {
        (Some(target), Some(translator)) if !result.text.is_empty() => {
            let target_name = resolve_target_language(target);
            let cached = {
                let cache = state.translation_cache.lock().await;
                cache.as_ref().and_then(|(t, tgt, tr)| {
                    (t == &result.text && tgt == &target_name).then(|| tr.clone())
                })
            };
            match cached {
                Some(tr) => Some(tr),
                None => {
                    let translated = translator
                        .translate(&result.text, &target_name)
                        .map_err(|e| ApiError::internal(format!("translation: {e}")))?;
                    *state.translation_cache.lock().await = Some((
                        result.text.clone(),
                        target_name,
                        translated.clone(),
                    ));
                    Some(translated)
                }
            }
        }
        _ => None,
    };
    drop(engines);

    info!(
        file = %filename,
        audio_s = format!("{:.2}", result.audio_seconds),
        elapsed_s = format!("{:.2}", result.elapsed_seconds),
        rtf = format!("{:.3}", result.rtf),
        gen_tokens = result.num_generated_tokens,
        translated = translation.is_some(),
        "transcription ok"
    );

    match response_format.as_str() {
        "text" => Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            translation.unwrap_or(result.text),
        )
            .into_response()),
        "verbose_json" => Ok(Json(VerboseTranscription {
            text: result.text,
            language: result.language,
            duration: result.audio_seconds,
            translation,
        })
        .into_response()),
        "json" => Ok(Json(TranscriptionJson {
            text: result.text,
            translation,
        })
        .into_response()),
        other => {
            warn!(response_format = %other, "unknown response_format; falling back to json");
            Ok(Json(TranscriptionJson {
                text: result.text,
                translation,
            })
            .into_response())
        }
    }
}

/// Streaming parameters from the official Qwen3-ASR recipe: the first
/// N chunks are decoded from scratch; afterwards the previous response is
/// kept as a forced prefix minus the last few (unstable) tokens.
const UNFIXED_CHUNK_NUM: usize = 2;
const UNFIXED_TOKEN_NUM: usize = 5;
/// Run a step once this much new audio has accumulated.
const LIVE_CHUNK_SECS: f32 = 1.0;
/// Commit the text and restart the buffer after this much audio, so the
/// re-encoding cost per step stays bounded.
const LIVE_SEGMENT_SECS: f32 = 15.0;

async fn live_ws(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = live_session(state, socket).await {
            warn!(error = %e, "live session ended with error");
        }
    })
}

/// Stateful live-transcription session over a WebSocket.
///
/// Protocol: the client streams raw PCM (16 kHz mono f32le) as binary frames
/// and JSON control messages (`{"type":"config","language":...}`,
/// `{"type":"stop"}`) as text frames. The server answers with
/// `{"type":"partial",...}` after every step and `{"type":"commit","text":..}`
/// when a segment is frozen.
async fn live_session(state: Arc<AppState>, mut socket: WebSocket) -> anyhow::Result<()> {
    let chunk_samples = (LIVE_CHUNK_SECS * SAMPLE_RATE as f32) as usize;
    let segment_samples = (LIVE_SEGMENT_SECS * SAMPLE_RATE as f32) as usize;
    let min_samples = SAMPLE_RATE as usize / 2; // 0.5 s before the first step

    let mut language: Option<String> = None;
    let mut buffer: Vec<f32> = Vec::new();
    let mut prev_tokens: Vec<u32> = Vec::new();
    let mut step_count = 0usize;
    let mut last_step_len = 0usize;
    let mut last_text = String::new();
    let mut stopping = false;
    let mut closed = false;

    info!("live session opened");

    while !closed {
        // Block briefly for the next frame, then drain whatever else queued
        // up while the previous step was running.
        let mut got_any = false;
        loop {
            let wait = if got_any { 2 } else { 100 };
            match tokio::time::timeout(Duration::from_millis(wait), socket.recv()).await {
                Ok(Some(Ok(msg))) => {
                    got_any = true;
                    match msg {
                        WsMessage::Binary(b) => {
                            buffer.extend(
                                b.chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                            );
                        }
                        WsMessage::Text(t) => {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                                match v.get("type").and_then(|x| x.as_str()) {
                                    Some("config") => {
                                        language = v
                                            .get("language")
                                            .and_then(|x| x.as_str())
                                            .filter(|s| !s.is_empty())
                                            .map(|s| s.to_string());
                                    }
                                    Some("stop") => stopping = true,
                                    _ => {}
                                }
                            }
                        }
                        WsMessage::Close(_) => {
                            closed = true;
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(Some(Err(_))) | Ok(None) => {
                    closed = true;
                    break;
                }
                Err(_) => break, // nothing pending right now
            }
        }

        let has_new = buffer.len() > last_step_len;
        let step_due = buffer.len() >= min_samples
            && (buffer.len() >= last_step_len + chunk_samples || (stopping && has_new));
        if !step_due {
            if stopping {
                // No new audio to process; still commit the last partial so
                // the client keeps the final text.
                if !last_text.is_empty() {
                    let commit =
                        serde_json::json!({ "type": "commit", "text": last_text });
                    let _ = socket
                        .send(WsMessage::Text(commit.to_string().into()))
                        .await;
                }
                break;
            }
            continue;
        }

        let forced: Vec<u32> = if step_count < UNFIXED_CHUNK_NUM {
            Vec::new()
        } else {
            let keep = prev_tokens.len().saturating_sub(UNFIXED_TOKEN_NUM);
            prev_tokens[..keep].to_vec()
        };

        let t0 = Instant::now();
        let step = {
            let mut engines = state.engines.lock().await;
            engines
                .asr
                .transcribe_stream_step(&buffer, language.as_deref(), &forced, 256)
        };
        let (tokens, tr) = match step {
            Ok(r) => r,
            Err(e) => {
                let _ = socket
                    .send(WsMessage::Text(
                        serde_json::json!({ "type": "error", "message": e.to_string() })
                            .to_string()
                            .into(),
                    ))
                    .await;
                break;
            }
        };
        prev_tokens = tokens;
        last_step_len = buffer.len();
        last_text = tr.text.clone();
        step_count += 1;

        let partial = serde_json::json!({
            "type": "partial",
            "text": tr.text,
            "language": tr.language,
            "audio_s": tr.audio_seconds,
            "step_ms": t0.elapsed().as_millis() as u64,
        });
        if socket
            .send(WsMessage::Text(partial.to_string().into()))
            .await
            .is_err()
        {
            break;
        }

        // Segment full (or session ending): freeze the text, restart buffer.
        if buffer.len() >= segment_samples || stopping {
            let commit = serde_json::json!({ "type": "commit", "text": tr.text });
            let _ = socket
                .send(WsMessage::Text(commit.to_string().into()))
                .await;
            buffer.clear();
            prev_tokens.clear();
            step_count = 0;
            last_step_len = 0;
            last_text.clear();
        }
        if stopping {
            break;
        }
    }

    // Graceful close: send the close frame, then drain until the peer
    // acknowledges so the TCP socket isn't reset mid-handshake.
    let _ = socket.send(WsMessage::Close(None)).await;
    loop {
        match tokio::time::timeout(Duration::from_millis(500), socket.recv()).await {
            Ok(Some(Ok(WsMessage::Close(_)))) | Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            Ok(Some(Ok(_))) => {}
        }
    }
    info!("live session closed");
    Ok(())
}

#[derive(Deserialize)]
struct TranslateRequest {
    text: String,
    target: String,
}

/// Standalone text translation endpoint. Lets the live UI fetch translations
/// in parallel with (instead of serially after) transcription steps.
async fn translate_text(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<TranslateRequest>,
) -> Result<Response, ApiError> {
    authorize(&headers, state.api_key.as_deref())?;
    if !state.has_translator {
        return Err(ApiError::bad_request(
            "translation not available: start the server with --translator-dir",
        ));
    }
    let text = req.text.trim().to_string();
    if text.is_empty() {
        return Ok(Json(serde_json::json!({ "translation": "" })).into_response());
    }
    let target_name = resolve_target_language(&req.target);

    let cached = {
        let cache = state.translation_cache.lock().await;
        cache.as_ref().and_then(|(t, tgt, tr)| {
            (t == &text && tgt == &target_name).then(|| tr.clone())
        })
    };
    let translation = match cached {
        Some(tr) => tr,
        None => {
            let translated = {
                let mut engines = state.engines.lock().await;
                let translator = engines
                    .translator
                    .as_mut()
                    .expect("checked has_translator above");
                translator
                    .translate(&text, &target_name)
                    .map_err(|e| ApiError::internal(format!("translation: {e}")))?
            };
            *state.translation_cache.lock().await =
                Some((text, target_name, translated.clone()));
            translated
        }
    };
    Ok(Json(serde_json::json!({ "translation": translation })).into_response())
}

/// Map a language code ("tr") to the English name used in the translation
/// prompt; unknown values pass through as-is (free-form target).
fn resolve_target_language(target: &str) -> String {
    qwen3_asr::config::resolve_language(Some(target))
        .ok()
        .flatten()
        .unwrap_or_else(|| target.to_string())
}

fn authorize(headers: &HeaderMap, expected: Option<&str>) -> Result<(), ApiError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return Err(ApiError {
            status: StatusCode::UNAUTHORIZED,
            message: "missing Authorization header".into(),
        });
    };
    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .unwrap_or(value);
    if token != expected {
        return Err(ApiError {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid API key".into(),
        });
    }
    Ok(())
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": match self.status {
                    StatusCode::UNAUTHORIZED => "invalid_request_error",
                    StatusCode::BAD_REQUEST => "invalid_request_error",
                    _ => "server_error",
                }
            }
        });
        (self.status, Json(body)).into_response()
    }
}
