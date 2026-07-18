use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

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

struct AppState {
    model: Mutex<AsrModel>,
    translator: Option<Mutex<Translator>>,
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
    let model = AsrModel::load(&args.model_dir)?;
    info!("model ready");

    let translator = match &args.translator_dir {
        Some(dir) => {
            let t = Translator::load(dir)?;
            Some(Mutex::new(t))
        }
        None => None,
    };

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        translator,
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
async function transcribe(blob, filename, language, translateTo) {
  const data = new FormData();
  data.append('file', blob, filename);
  data.append('model', 'cibs');
  data.append('response_format', 'verbose_json');
  if (language) data.append('language', language);
  if (translateTo) data.append('translate_to', translateTo);
  const r = await fetch('/v1/audio/transcriptions', { method: 'POST', body: data });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error?.message || r.status);
  return j;
}

// Segmented streaming: MediaRecorder chunks accumulate and the recording so
// far is re-transcribed every interval. To keep the per-step cost (and thus
// the latency) bounded, the recorder is restarted every SEGMENT_MS; finished
// segments are frozen and prepended to the display. Translation runs in a
// separate loop against /v1/translate so it never blocks transcription.
const SEGMENT_MS = 15000;
const STEP_MS = 1500;
const TR_MS = 1200;
const micBtn = document.getElementById('mic');
const asrOut = document.getElementById('asrOut');
const trOut = document.getElementById('trOut');
const stat = document.getElementById('stat');
const dot = document.getElementById('dot');
let rec = null, stream = null, chunks = [], timer = null, trTimer = null;
let busy = false, trBusy = false, lastTranslated = null;
let segStart = 0;
let segTexts = [], segTrs = [];   // committed (frozen) segments
let curText = '', curTr = '';     // current segment, still changing

function extFor(mime) {
  if (!mime) return 'webm';
  if (mime.includes('mp4')) return 'mp4';
  if (mime.includes('ogg')) return 'ogg';
  return 'webm';
}

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
// transcription loop.
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

function startSegment() {
  chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
  rec.start(500);
  segStart = performance.now();
}

async function liveStep(final = false) {
  if (busy || chunks.length === 0) return;
  busy = true;
  // Segment full: freeze what we have and restart the recorder so the next
  // request only carries fresh audio (bounds per-step latency).
  const rollover = rec && (performance.now() - segStart) > SEGMENT_MS;
  if (rollover) {
    rec.stop();
    await new Promise(r => setTimeout(r, 150)); // flush last chunk
  }
  const mime = rec ? rec.mimeType : 'audio/webm';
  const blob = new Blob(chunks, { type: mime });
  if (rollover) startSegment();
  try {
    const t0 = performance.now();
    const j = await transcribe(blob, 'live.' + extFor(mime),
                               document.getElementById('srcLang').value, null);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    curText = j.text || '';
    if (rollover || final) {
      const idx = segTexts.length;
      segTexts.push(curText);
      segTrs.push(curTr || '…');
      finalizeSegment(idx, curText);
      curText = ''; curTr = ''; lastTranslated = null;
    }
    render();
    stat.textContent = (j.duration?.toFixed(0) || '?') + ' sn ses · adım ' +
                       secs + ' sn' + (j.language ? ' · ' + j.language : '');
  } catch (err) {
    stat.textContent = 'hata: ' + err.message;
  } finally {
    busy = false;
    if (final) stat.textContent += ' · bitti';
  }
}

micBtn.addEventListener('click', async () => {
  if (rec) { // stop
    clearInterval(timer); timer = null;
    clearInterval(trTimer); trTimer = null;
    const r = rec;
    rec = null;
    r.stop();
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    micBtn.textContent = 'Dinlemeye başla';
    micBtn.classList.remove('rec');
    dot.style.visibility = 'hidden';
    setTimeout(() => liveStep(true), 400); // final pass with remaining audio
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    trOut.textContent = 'Mikrofon izni alınamadı: ' + err.message;
    return;
  }
  segTexts = []; segTrs = []; curText = ''; curTr = ''; lastTranslated = null;
  startSegment();
  timer = setInterval(liveStep, STEP_MS);
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

    if translate_to.is_some() && state.translator.is_none() {
        return Err(ApiError::bad_request(
            "translation not available: start the server with --translator-dir",
        ));
    }

    let max_tokens = state.max_tokens;
    let result = {
        let mut model = state.model.lock().await;
        model
            .transcribe_bytes(&bytes, &filename, language.as_deref(), max_tokens)
            .map_err(|e| ApiError::internal(e.to_string()))?
    };

    // Optional local translation of the transcript.
    let translation = match (&translate_to, &state.translator) {
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
                    let translated = {
                        let mut tr = translator.lock().await;
                        tr.translate(&result.text, &target_name)
                            .map_err(|e| ApiError::internal(format!("translation: {e}")))?
                    };
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
    let Some(translator) = &state.translator else {
        return Err(ApiError::bad_request(
            "translation not available: start the server with --translator-dir",
        ));
    };
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
                let mut tr = translator.lock().await;
                tr.translate(&text, &target_name)
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
