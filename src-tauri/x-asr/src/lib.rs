//! CPU X-ASR inference. Inputs are mono 16 kHz normalized f32 samples.
//! Offline decoding is independent, pause-aware segments of at most 30 seconds;
//! it does not claim unlimited full-utterance context.

use anyhow::{anyhow, bail, ensure, Context, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::LazyLock;

const SAMPLE_RATE: usize = 16_000;
const MIN_OFFLINE_SAMPLES: usize = SAMPLE_RATE / 10;
const MAX_OFFLINE_SAMPLES: usize = 30 * SAMPLE_RATE;
const ERROR_CAPACITY: usize = 4096;

type Native = c_void;

extern "C" {
    fn handy_xasr_ort_api(
        version: u32,
        api: *mut *const c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_load(
        directory: *const c_char,
        offline: c_int,
        model: *mut *mut Native,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_destroy(model: *mut Native, error: *mut c_char, capacity: usize) -> c_int;
    fn handy_xasr_offline(
        model: *mut Native,
        samples: *const f32,
        count: i32,
        text: *mut *const c_char,
        length: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_start(
        model: *mut Native,
        stream: *mut *mut Native,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_feed(
        stream: *mut Native,
        samples: *const f32,
        count: i32,
        text: *mut *const c_char,
        length: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_finish(
        stream: *mut Native,
        text: *mut *const c_char,
        length: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn handy_xasr_cancel(stream: *mut Native, error: *mut c_char, capacity: usize) -> c_int;
}

fn checked(call: impl FnOnce(*mut c_char, usize) -> c_int) -> Result<()> {
    let mut error = [0 as c_char; ERROR_CAPACITY];
    let status = call(error.as_mut_ptr(), error.len());
    if status != 0 {
        // The shim always terminates this caller-owned, zero-initialized buffer.
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        bail!("X-ASR native error ({status}): {message}");
    }
    Ok(())
}

/// Initialize ort with the exact shared runtime used by the native recognizers.
/// Must run before any other application ort API (including VAD/model loading).
/// Repeated calls are harmless; a previously initialized different backend is an error.
pub fn init_runtime() -> Result<()> {
    static INITIALIZED: LazyLock<std::result::Result<(), String>> = LazyLock::new(|| {
        let initialize = || -> Result<()> {
            let mut api = ptr::null();
            checked(|error, capacity| unsafe {
                handy_xasr_ort_api(ort::sys::ORT_API_VERSION, &mut api, error, capacity)
            })?;
            ensure!(!api.is_null(), "X-ASR returned an empty ORT API");
            // ORT owns the static function table for the life of the process.
            // ort::set_api takes a value copy, not ownership of the original table.
            let api = unsafe { ptr::read(api.cast::<ort::sys::OrtApi>()) };
            ensure!(
                ort::set_api(api),
                "ort was initialized before the shared X-ASR runtime"
            );
            Ok(())
        };
        initialize().map_err(|error| format!("{error:#}"))
    });
    INITIALIZED
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))
        .copied()
}

/// Select the matched streaming or full-context offline model export.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Streaming,
    Offline,
}

/// A reusable CPU recognizer; utterance state never survives a transcription run.
pub struct XAsrModel {
    native: NonNull<Native>,
    mode: Mode,
}

// Native ORT sessions may move between threads. All operations require exclusive
// access; a stream borrows its model and cannot outlive it. Deliberately not Sync.
unsafe impl Send for XAsrModel {}

impl XAsrModel {
    /// Load a matched model directory after the caller has verified its artifacts.
    pub fn load(directory: &Path, mode: Mode) -> Result<Self> {
        init_runtime()?;
        let required = match mode {
            Mode::Streaming => [
                "encoder-480ms.onnx",
                "decoder-480ms.onnx",
                "joiner-480ms.onnx",
                "tokens.txt",
            ],
            Mode::Offline => [
                "encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
                "tokens.txt",
            ],
        };
        for file in required {
            ensure!(
                directory.join(file).is_file(),
                "missing X-ASR component: {}",
                directory.join(file).display()
            );
        }
        let directory = CString::new(
            directory
                .to_str()
                .context("X-ASR model path is not UTF-8")?,
        )?;
        let mut native = ptr::null_mut();
        checked(|error, capacity| unsafe {
            handy_xasr_load(
                directory.as_ptr(),
                (mode == Mode::Offline) as c_int,
                &mut native,
                error,
                capacity,
            )
        })?;
        Ok(Self {
            native: NonNull::new(native).context("X-ASR returned an empty model")?,
            mode,
        })
    }

    /// Transcribe mono 16 kHz PCM. Offline input is padded or segmented to the
    /// model's supported length range without discarding samples.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        if self.mode == Mode::Streaming {
            let mut stream = self.start_stream()?;
            stream.feed(samples)?;
            return stream.finish();
        }

        let mut remaining = samples;
        let mut text = String::new();
        while !remaining.is_empty() {
            let end = offline_segment_len(remaining);
            let segment = &remaining[..end];
            let mut padded;
            let input = if segment.len() < MIN_OFFLINE_SAMPLES {
                padded = [0.0; MIN_OFFLINE_SAMPLES];
                padded[..segment.len()].copy_from_slice(segment);
                &padded[..]
            } else {
                segment
            };
            let mut raw = ptr::null();
            let mut length = 0;
            checked(|error, capacity| unsafe {
                handy_xasr_offline(
                    self.native.as_ptr(),
                    input.as_ptr(),
                    input.len() as i32,
                    &mut raw,
                    &mut length,
                    error,
                    capacity,
                )
            })?;
            let segment_text = native_text(raw, length)?;
            append_segment(&mut text, &segment_text);
            remaining = &remaining[end..];
        }
        Ok(text)
    }

    /// Begin an independent utterance. Offline models reject this operation.
    pub fn start_stream(&mut self) -> Result<XAsrStream<'_>> {
        ensure!(
            self.mode == Mode::Streaming,
            "offline X-ASR does not support streaming"
        );
        let mut native = ptr::null_mut();
        checked(|error, capacity| unsafe {
            handy_xasr_start(self.native.as_ptr(), &mut native, error, capacity)
        })?;
        Ok(XAsrStream {
            native: NonNull::new(native).context("X-ASR returned an empty stream")?,
            last_text: String::new(),
            failed: false,
            _model: PhantomData,
        })
    }
}

impl Drop for XAsrModel {
    fn drop(&mut self) {
        if let Err(error) = checked(|error, capacity| unsafe {
            handy_xasr_destroy(self.native.as_ptr(), error, capacity)
        }) {
            log::error!("Could not destroy X-ASR model: {error:#}");
        }
    }
}

/// Exclusive utterance state. Dropping it cancels without unloading the model.
pub struct XAsrStream<'model> {
    native: NonNull<Native>,
    last_text: String,
    failed: bool,
    _model: PhantomData<&'model mut XAsrModel>,
}

impl XAsrStream<'_> {
    /// Return only a changed complete hypothesis, not a token delta.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<String>> {
        ensure!(
            !self.failed,
            "X-ASR stream failed; drop it before starting another"
        );
        let mut raw = ptr::null();
        let mut length = 0;
        // Bound the native feature queue even when transcribing a whole recording.
        for chunk in samples.chunks(SAMPLE_RATE) {
            let mut changed_text = ptr::null();
            let mut changed_length = 0;
            if let Err(error) = checked(|error, capacity| unsafe {
                handy_xasr_feed(
                    self.native.as_ptr(),
                    chunk.as_ptr(),
                    chunk.len() as i32,
                    &mut changed_text,
                    &mut changed_length,
                    error,
                    capacity,
                )
            }) {
                self.failed = true;
                return Err(error);
            }
            if !changed_text.is_null() {
                raw = changed_text;
                length = changed_length;
            }
        }
        if raw.is_null() {
            return Ok(None);
        }
        let text = native_text(raw, length)?;
        if text == self.last_text {
            Ok(None)
        } else {
            self.last_text.clone_from(&text);
            Ok(Some(text))
        }
    }

    /// Append the model's required 1.5s tail, finish input, and drain the decoder.
    pub fn finish(self) -> Result<String> {
        ensure!(
            !self.failed,
            "X-ASR stream failed; its result is not complete"
        );
        let mut raw = ptr::null();
        let mut length = 0;
        checked(|error, capacity| unsafe {
            handy_xasr_finish(self.native.as_ptr(), &mut raw, &mut length, error, capacity)
        })?;
        native_text(raw, length)
    }
}

impl Drop for XAsrStream<'_> {
    fn drop(&mut self) {
        if let Err(error) = checked(|error, capacity| unsafe {
            handy_xasr_cancel(self.native.as_ptr(), error, capacity)
        }) {
            log::error!("Could not destroy X-ASR stream: {error:#}");
        }
    }
}

fn native_text(raw: *const c_char, length: usize) -> Result<String> {
    ensure!(!raw.is_null(), "X-ASR returned an empty text pointer");
    // Borrowed only until the next native operation; normalize into owned text now.
    let bytes = unsafe { std::slice::from_raw_parts(raw.cast::<u8>(), length) };
    Ok(normalize_text(
        std::str::from_utf8(bytes).context("X-ASR returned invalid UTF-8")?,
    ))
}

fn offline_segment_len(samples: &[f32]) -> usize {
    if samples.len() <= MAX_OFFLINE_SAMPLES {
        return samples.len();
    }
    // Leave at least 100ms for the next segment, even for a 30s+1-sample input.
    let limit = MAX_OFFLINE_SAMPLES.min(samples.len() - MIN_OFFLINE_SAMPLES);
    let search_start = limit - 5 * SAMPLE_RATE;
    const FRAME: usize = SAMPLE_RATE / 50; // 20ms
    const PAUSE: usize = SAMPLE_RATE / 5; // 200ms
    let mut quiet_start = None;
    let mut best_pause = 0;
    let mut cut = limit;
    for start in (search_start..limit).step_by(FRAME) {
        let end = (start + FRAME).min(limit);
        let energy = samples[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32;
        if energy <= 0.0001 {
            let quiet = *quiet_start.get_or_insert(start);
            let duration = end - quiet;
            if duration >= PAUSE && duration >= best_pause {
                best_pause = duration;
                cut = quiet + duration / 2;
            }
        } else {
            quiet_start = None;
        }
    }
    cut
}

fn cjk_or_punctuation(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
        || "，。！？；：、（）《》〈〉【】「」『』“”‘’".contains(c)
}

fn remove_space(left: char, right: char) -> bool {
    (cjk_or_punctuation(left) && cjk_or_punctuation(right)) || ",.!?;:%)]}".contains(right)
}

// The author's _normalize_cjk_spacing recipe, with text_format="none".
// Whitespace at Chinese/English boundaries is retained; English case is untouched.
fn normalize_text(text: &str) -> String {
    let text = text.trim();
    let mut output = String::with_capacity(text.len());
    let mut previous = None;
    let mut whitespace_start = None;
    for (offset, c) in text.char_indices() {
        if c.is_whitespace() {
            whitespace_start.get_or_insert(offset);
            continue;
        }
        if let Some(start) = whitespace_start.take() {
            if previous.is_some_and(|left| !remove_space(left, c)) {
                output.push_str(&text[start..offset]);
            }
        }
        output.push(c);
        previous = Some(c);
    }
    output
}

fn append_segment(text: &mut String, segment: &str) {
    if segment.is_empty() {
        return;
    }
    if let (Some(left), Some(right)) = (text.chars().next_back(), segment.chars().next()) {
        if !remove_space(left, right) {
            text.push(' ');
        }
    }
    text.push_str(segment);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_preserves_tail_and_continuous_audio() {
        for length in [MAX_OFFLINE_SAMPLES + 1, 2 * MAX_OFFLINE_SAMPLES + 731] {
            let audio: Vec<f32> = (0..length).map(|i| 0.2 + (i % 97) as f32 / 200.0).collect();
            let mut rest = audio.as_slice();
            let mut reconstructed = Vec::new();
            while !rest.is_empty() {
                let end = offline_segment_len(rest);
                assert!((MIN_OFFLINE_SAMPLES..=MAX_OFFLINE_SAMPLES).contains(&end));
                reconstructed.extend_from_slice(&rest[..end]);
                rest = &rest[end..];
            }
            assert_eq!(audio, reconstructed);
        }
    }

    #[test]
    fn segmentation_prefers_a_pause_to_a_hard_cut() {
        let mut audio = vec![0.5; 40 * SAMPLE_RATE];
        audio[28 * SAMPLE_RATE..29 * SAMPLE_RATE].fill(0.0);
        let cut = offline_segment_len(&audio);
        assert!((28 * SAMPLE_RATE..29 * SAMPLE_RATE).contains(&cut));
        assert_eq!(offline_segment_len(&[0.5; 1]), 1);
        assert_eq!(offline_segment_len(&[]), 0);
    }

    #[test]
    fn model_spacing_preserves_mixed_language_case_and_segment_boundaries() {
        assert_eq!(
            normalize_text(" 你 好 ， 世 界 ！ OpenAI API is Ready . "),
            "你好，世界！ OpenAI API is Ready."
        );
        assert_eq!(
            normalize_text("中文 English\tWords 中文"),
            "中文 English\tWords 中文"
        );
        let mut joined = String::from("你好，");
        append_segment(&mut joined, "世界");
        append_segment(&mut joined, "OpenAI");
        append_segment(&mut joined, "Works.");
        assert_eq!(joined, "你好，世界 OpenAI Works.");
    }
}
