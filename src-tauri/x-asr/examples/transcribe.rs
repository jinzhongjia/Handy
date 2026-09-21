use anyhow::{bail, ensure, Context, Result};
use handy_x_asr::{init_runtime, Mode, XAsrModel};
use std::path::Path;
use std::time::Instant;

// cargo run -p handy-x-asr --example transcribe -- offline MODEL_DIR MONO_16K.wav [REPEATS]
// Use streaming instead of offline to print changed full hypotheses and final text.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "usage: transcribe streaming|offline MODEL_DIR MONO_16K.wav [REPEATS]"
    );
    let mode = match args[1].as_str() {
        "streaming" => Mode::Streaming,
        "offline" => Mode::Offline,
        value => bail!("unknown mode: {value}"),
    };
    let repeats = args
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let mut reader = hound::WavReader::open(&args[3]).context("open WAV")?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == 16000,
        "expected mono 16kHz WAV"
    );
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let samples = samples.repeat(repeats);
    init_runtime()?;
    let load = Instant::now();
    let mut model = XAsrModel::load(Path::new(&args[2]), mode)?;
    eprintln!(
        "loaded={:.3}s samples={} audio={:.3}s",
        load.elapsed().as_secs_f64(),
        samples.len(),
        samples.len() as f64 / 16000.0
    );
    let start = Instant::now();
    let text = if mode == Mode::Streaming {
        let mut stream = model.start_stream()?;
        for chunk in samples.chunks(1600) {
            if let Some(text) = stream.feed(chunk)? {
                println!("partial: {text}");
            }
        }
        stream.finish()?
    } else {
        model.transcribe(&samples)?
    };
    println!("final: {text}");
    eprintln!("decode={:.3}s", start.elapsed().as_secs_f64());
    // Empty input and cancellation must leave the same recognizer reusable.
    ensure!(
        model.transcribe(&[])?.is_empty(),
        "empty input produced text"
    );
    if mode == Mode::Streaming {
        let mut cancelled = model.start_stream()?;
        cancelled.feed(&samples[..samples.len().min(1600)])?;
        drop(cancelled);
        ensure!(
            model.start_stream()?.finish()?.is_empty(),
            "empty stream produced text"
        );
    }
    Ok(())
}
