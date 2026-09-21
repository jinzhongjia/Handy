//! Matched X-ASR bundles. Recipes are private, pinned trust anchors, not alternate quants.
use super::download::{HttpDownloadEvent, HttpDownloadOutcome};
use super::{
    DiskStatus, DownloadCleanup, DownloadProgress, EngineType, ModelInfo, ModelManager, ModelSource,
};
use anyhow::{ensure, Context, Result};
use bzip2::read::BzDecoder;
use hf_hub::api::tokio::CancellationToken;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use tauri::Emitter;

struct BundleFile {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

struct BundleArchive {
    url: &'static str,
    root: &'static str,
    size: u64,
    sha256: &'static str,
}

struct BundleRecipe {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    engine: EngineType,
    files: &'static [BundleFile],
    archive: Option<BundleArchive>,
}

const STREAMING_BASE: &str = "https://huggingface.co/GilgameshWind/X-ASR-zh-en/resolve/689ff18c584d29910da37b6fe904db0c1489c9d1/deployment/models/chunk-480ms-model";
const TOKENS: BundleFile = BundleFile {
    name: "tokens.txt",
    size: 58806,
    sha256: "b818a60878b9aae978cbb8ad594acbd403d76d1af2e31ef4197c84e2dbdba27c",
};
const RECIPES: [BundleRecipe; 2] = [
    BundleRecipe {
        id: "x-asr-zh-en-streaming-480ms",
        name: "X-ASR Chinese–English Streaming (480 ms)",
        description: "Chinese and English with live transcription preview. Runs locally on CPU; recognizes both languages automatically.",
        engine: EngineType::XAsrStreaming,
        files: &[
            BundleFile { name: "encoder-480ms.onnx", size: 592968361, sha256: "0c3454033d249081df124ddcd7adaf3deca07d0b999b26f2ee5d2475d37abc74" },
            BundleFile { name: "decoder-480ms.onnx", size: 11309084, sha256: "3658368d274a5d5fd39a7ac20c46bed0ad9cfea1f0feddef30d5d89797c1f499" },
            BundleFile { name: "joiner-480ms.onnx", size: 10260467, sha256: "03781c98165a2385024c9cecdd2b6b13310d81db23a62c7da420782c2915cf81" },
            TOKENS,
        ],
        archive: None,
    },
    BundleRecipe {
        id: "x-asr-zh-en-offline-int8",
        name: "X-ASR Chinese–English Offline (INT8)",
        description: "Chinese and English with punctuation. A genuine non-streaming INT8 model running locally on CPU. Long recordings are transcribed in segments of up to 30 seconds.",
        engine: EngineType::XAsrOffline,
        files: &[
            BundleFile { name: "encoder-epoch-99-avg-1.int8.onnx", size: 161015713, sha256: "7f6aa62056efd8af9da13e0faa81cd3f284d2fb2e3b63de56fd2dfd3450910dc" },
            BundleFile { name: "decoder-epoch-99-avg-1.onnx", size: 11309084, sha256: "72f47405d3c1033bebccbef82f90071e7b4ba3e71b9c986f2b74244b25723aed" },
            BundleFile { name: "joiner-epoch-99-avg-1.int8.onnx", size: 2581422, sha256: "aedb7fa697b2ab43f20499826fff7c997eea7d67db77be97769aeeeb726e63b3" },
            TOKENS,
        ],
        archive: Some(BundleArchive {
            url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2",
            root: "sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03",
            size: 136396739,
            // The dated upstream URL was replaced on June 29; the hash pins that repaired archive.
            sha256: "5d02c36d7b44e886b7c8f0d8e051f8713acab96c264bb6ef9e718be39a6a2224",
        }),
    },
];

#[derive(Debug, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    modified: SystemTime,
    created: Option<SystemTime>,
}

#[derive(Default)]
pub(super) struct BundleState {
    // A resumed request waits until its cancelled predecessor has stopped touching disk.
    download: tokio::sync::Mutex<()>,
    cancellation_epoch: AtomicU64,
    revision: AtomicU64,
    verified: Mutex<Option<Vec<FileStamp>>>,
}

impl BundleState {
    async fn acquire_download(&self) -> Option<(tokio::sync::MutexGuard<'_, ()>, u64)> {
        // Queued retries belong to the cancellation epoch in which they began.
        let epoch = self.cancellation_epoch.load(Ordering::Acquire);
        let guard = self.download.lock().await;
        self.is_current_request(epoch).then_some((guard, epoch))
    }

    fn is_current_request(&self, epoch: u64) -> bool {
        self.cancellation_epoch.load(Ordering::Acquire) == epoch
    }

    fn invalidate_snapshot(&self) {
        // Status writers hold the registry lock when advancing this revision.
        self.revision.fetch_add(1, Ordering::Release);
    }

    pub(super) fn apply_snapshot(
        &self,
        model: &mut ModelInfo,
        revision: u64,
        status: &DiskStatus,
        downloading: bool,
    ) {
        if self.revision.load(Ordering::Acquire) == revision {
            model.is_downloaded = status.is_downloaded;
            model.partial_size = status.partial_size;
            model.is_downloading = downloading;
        }
    }
}

fn bundle_recipe(id: &str) -> Result<(usize, &'static BundleRecipe)> {
    RECIPES
        .iter()
        .enumerate()
        .find(|(_, recipe)| recipe.id == id)
        .with_context(|| format!("Unknown model bundle: {id}"))
}

fn transfer_size(recipe: &BundleRecipe) -> u64 {
    recipe.archive.as_ref().map_or_else(
        || recipe.files.iter().map(|file| file.size).sum(),
        |archive| archive.size,
    )
}

pub(super) fn register_models(models: &mut HashMap<String, ModelInfo>) {
    for recipe in &RECIPES {
        let streaming = matches!(recipe.engine, EngineType::XAsrStreaming);
        models.insert(
            recipe.id.to_string(),
            ModelInfo {
                id: recipe.id.to_string(),
                name: recipe.name.to_string(),
                description: recipe.description.to_string(),
                filename: recipe.id.to_string(),
                source: ModelSource::Bundle,
                size_mb: transfer_size(recipe) / (1024 * 1024),
                is_downloaded: false,
                is_downloading: false,
                partial_size: 0,
                is_directory: true,
                engine_type: recipe.engine.clone(),
                // No comparable accuracy/speed benchmark is available for these models.
                accuracy_score: 0.0,
                speed_score: 0.0,
                supports_translation: false,
                is_recommended: false,
                supported_languages: vec!["zh".to_string(), "en".to_string()],
                supports_language_selection: false,
                is_custom: false,
                supports_streaming: streaming,
                // Automatic bilingual recognition, not a separate language-identification result.
                supports_language_detection: true,
            },
        );
    }
}

fn component_stamps(dir: &Path, files: &[BundleFile]) -> Result<Vec<FileStamp>> {
    ensure!(
        fs::symlink_metadata(dir)?.is_dir(),
        "Bundle is not a real directory: {}",
        dir.display()
    );
    files
        .iter()
        .map(|file| {
            let path = dir.join(file.name);
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("Missing bundle component: {}", path.display()))?;
            ensure!(
                metadata.is_file() && metadata.len() == file.size,
                "Incomplete bundle component: {}",
                path.display()
            );
            Ok(FileStamp {
                size: metadata.len(),
                modified: metadata.modified()?,
                created: metadata.created().ok(),
            })
        })
        .collect()
}

fn verify_components(dir: &Path, files: &[BundleFile]) -> Result<Vec<FileStamp>> {
    let before = component_stamps(dir, files)?;
    for file in files {
        ensure!(
            ModelManager::compute_sha256(&dir.join(file.name))? == file.sha256,
            "Mismatched bundle component: {}",
            file.name
        );
    }
    ensure!(
        before == component_stamps(dir, files)?,
        "Bundle changed during verification"
    );
    Ok(before)
}

fn remove_path(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(path)?,
        Ok(_) => fs::remove_file(path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn ensure_staging_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    ensure!(
        fs::symlink_metadata(path)?.is_dir(),
        "Invalid bundle staging directory: {}",
        path.display()
    );
    Ok(())
}

fn saved_size(path: &Path, limit: u64) -> u64 {
    fs::symlink_metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file())
        .map_or(0, |metadata| metadata.len().min(limit))
}

// Never unpack arbitrary tar paths. Only exact regular components under the pinned
// archive root may be written, and each is size/hash checked while copying.
fn extract_archive(
    archive_path: &Path,
    output: &Path,
    recipe: &BundleRecipe,
    cancel: &CancellationToken,
) -> Result<bool> {
    let archive_spec = recipe.archive.as_ref().context("Bundle has no archive")?;
    let mut archive = tar::Archive::new(BzDecoder::new(File::open(archive_path)?));
    let mut found = HashSet::new();
    for entry in archive.entries()? {
        if cancel.is_cancelled() {
            return Ok(false);
        }
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        ensure!(
            path.components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir)),
            "Unsafe bundle archive path: {}",
            path.display()
        );
        let relative = path
            .strip_prefix(archive_spec.root)
            .with_context(|| format!("Unexpected bundle archive root: {}", path.display()))?;
        let kind = entry.header().entry_type();
        ensure!(
            kind.is_file() || kind.is_dir(),
            "Unsupported bundle archive entry: {}",
            path.display()
        );
        if kind.is_dir() {
            continue;
        }
        let Some(file) = recipe
            .files
            .iter()
            .find(|file| relative == Path::new(file.name))
        else {
            continue;
        };
        ensure!(
            found.insert(file.name),
            "Duplicate bundle component: {}",
            file.name
        );
        ensure!(
            entry.size() == file.size,
            "Wrong size for archived component: {}",
            file.name
        );
        let mut target = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(file.name))?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            if cancel.is_cancelled() {
                return Ok(false);
            }
            let count = entry.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            target.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
        }
        target.sync_all()?;
        ensure!(
            format!("{:x}", hash.finalize()) == file.sha256,
            "Mismatched archived component: {}",
            file.name
        );
    }
    ensure!(
        found.len() == recipe.files.len(),
        "Bundle archive is missing required components"
    );
    component_stamps(output, recipe.files)?;
    Ok(!cancel.is_cancelled())
}

impl ModelManager {
    pub(super) fn bundle_disk_statuses(&self) -> [(&'static str, u64, DiskStatus); 2] {
        std::array::from_fn(|index| {
            let id = RECIPES[index].id;
            let revision = self.bundle_states[index].revision.load(Ordering::Acquire);
            let is_downloaded = self.verified_bundle_path(id).is_ok();
            (
                id,
                revision,
                DiskStatus {
                    is_downloaded,
                    partial_size: if is_downloaded {
                        0
                    } else {
                        self.bundle_partial_size(id)
                    },
                    ..DiskStatus::default()
                },
            )
        })
    }

    pub(super) fn verified_bundle_path(&self, id: &str) -> Result<PathBuf> {
        let (index, recipe) = bundle_recipe(id)?;
        let path = self.models_dir.join(recipe.id);
        let stamps = component_stamps(&path, recipe.files)?;
        let mut cached = self.bundle_states[index].verified.lock();
        if cached.as_ref() != Some(&stamps) {
            *cached = None;
            *cached = Some(verify_components(&path, recipe.files)?);
        }
        Ok(path)
    }

    pub(super) fn bundle_partial_size(&self, id: &str) -> u64 {
        let Ok((_, recipe)) = bundle_recipe(id) else {
            return 0;
        };
        let staging = self.models_dir.join(format!("{}.partial", recipe.id));
        if let Some(archive) = &recipe.archive {
            saved_size(&staging.join("archive.tar.bz2"), archive.size)
        } else {
            recipe
                .files
                .iter()
                .map(|file| saved_size(&staging.join(file.name), file.size))
                .sum()
        }
    }

    fn emit_bundle_progress(&self, id: &str, downloaded: u64, total: u64) {
        let _ = self.app_handle.emit(
            "model-download-progress",
            DownloadProgress {
                model_id: id.to_string(),
                downloaded,
                total,
                percentage: downloaded as f64 / total as f64 * 100.0,
            },
        );
    }

    async fn download_bundle_file(
        &self,
        recipe: &BundleRecipe,
        url: &str,
        path: &Path,
        file: &BundleFile,
        offset: u64,
        cancel: &CancellationToken,
    ) -> Result<bool> {
        // Never follow a dropped-in symlink when resuming into Handy's staging directory.
        if let Ok(metadata) = fs::symlink_metadata(path) {
            ensure!(
                metadata.is_file(),
                "Invalid partial bundle component: {}",
                path.display()
            );
        }
        let total = transfer_size(recipe);
        let result = Self::download_http_resumable_with_events(
            recipe.id,
            url,
            path,
            Some(file.size),
            Some(file.sha256),
            cancel,
            &|event| match event {
                HttpDownloadEvent::Progress(progress) => {
                    self.emit_bundle_progress(recipe.id, offset + progress.downloaded, total)
                }
                HttpDownloadEvent::VerificationStarted => {
                    let _ = self
                        .app_handle
                        .emit("model-verification-started", recipe.id);
                }
                HttpDownloadEvent::VerificationCompleted => {
                    let _ = self
                        .app_handle
                        .emit("model-verification-completed", recipe.id);
                }
            },
        )
        .await?;
        if matches!(result, HttpDownloadOutcome::Cancelled) || cancel.is_cancelled() {
            return Ok(false);
        }
        self.emit_bundle_progress(recipe.id, offset + file.size, total);
        Ok(true)
    }

    async fn acquire_bundle(
        &self,
        recipe: &'static BundleRecipe,
        cancel: &CancellationToken,
    ) -> Result<bool> {
        let staging = self.models_dir.join(format!("{}.partial", recipe.id));
        let extracted = self.models_dir.join(format!("{}.extracting", recipe.id));
        ensure_staging_dir(&staging)?;
        let publish_from = if let Some(archive) = &recipe.archive {
            let file = BundleFile {
                name: "archive.tar.bz2",
                size: archive.size,
                sha256: archive.sha256,
            };
            let archive_path = staging.join(file.name);
            if !self
                .download_bundle_file(recipe, archive.url, &archive_path, &file, 0, cancel)
                .await?
            {
                return Ok(false);
            }
            remove_path(&extracted)?;
            fs::create_dir(&extracted)?;
            let _ = self.app_handle.emit("model-extraction-started", recipe.id);
            let output = extracted.clone();
            let token = cancel.clone();
            let result = tokio::task::spawn_blocking(move || {
                extract_archive(&archive_path, &output, recipe, &token)
            })
            .await
            .context("Bundle extraction task failed")
            .and_then(|result| result);
            match result {
                Ok(true) => {}
                Ok(false) => {
                    remove_path(&extracted)?;
                    return Ok(false);
                }
                Err(error) => {
                    let _ = remove_path(&extracted);
                    let _ = self.app_handle.emit(
                        "model-extraction-failed",
                        serde_json::json!({ "model_id": recipe.id, "error": error.to_string() }),
                    );
                    return Err(error);
                }
            }
            extracted.clone()
        } else {
            for file in recipe.files {
                if cancel.is_cancelled() {
                    return Ok(false);
                }
                let offset = recipe
                    .files
                    .iter()
                    .filter(|other| other.name != file.name)
                    .map(|other| saved_size(&staging.join(other.name), other.size))
                    .sum();
                let url = format!("{STREAMING_BASE}/{}", file.name);
                if !self
                    .download_bundle_file(
                        recipe,
                        &url,
                        &staging.join(file.name),
                        file,
                        offset,
                        cancel,
                    )
                    .await?
                {
                    return Ok(false);
                }
            }
            staging.clone()
        };
        if cancel.is_cancelled() {
            return Ok(false);
        }
        // Every component was hashed by the transport or extractor; do not read
        // the entire 600 MB bundle a second time just to cache its identity.
        let stamps = component_stamps(&publish_from, recipe.files)?;
        let destination = self.models_dir.join(recipe.id);
        remove_path(&destination)?;
        fs::rename(&publish_from, &destination)?;
        let (index, _) = bundle_recipe(recipe.id)?;
        *self.bundle_states[index].verified.lock() = Some(stamps);
        if recipe.archive.is_some() {
            let _ = remove_path(&staging);
            let _ = self
                .app_handle
                .emit("model-extraction-completed", recipe.id);
        }
        Ok(true)
    }

    pub(super) async fn download_bundle_model(&self, model: &ModelInfo) -> Result<()> {
        let (index, recipe) = bundle_recipe(&model.id)?;
        let Some((_download, epoch)) = self.bundle_states[index].acquire_download().await else {
            return Ok(());
        };
        if self.verified_bundle_path(recipe.id).is_ok() {
            self.update_download_status()?;
            return Ok(());
        }
        let cancel = CancellationToken::new();
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(model.id.clone(), cancel.clone());
        let cleanup = DownloadCleanup {
            available_models: &self.available_models,
            cancel_flags: &self.cancel_flags,
            model_id: model.id.clone(),
            disarmed: false,
        };
        // Cancellation may race active-token registration after the lock wait.
        if !self.bundle_states[index].is_current_request(epoch) {
            cancel.cancel();
        }
        if let Some(info) = self.available_models.lock().unwrap().get_mut(recipe.id) {
            info.is_downloading = !cancel.is_cancelled();
            self.bundle_states[index].invalidate_snapshot();
        }
        let result = if cancel.is_cancelled() {
            Ok(false)
        } else {
            self.emit_bundle_progress(
                recipe.id,
                self.bundle_partial_size(recipe.id),
                transfer_size(recipe),
            );
            self.acquire_bundle(recipe, &cancel).await
        };
        drop(cleanup);
        // Refresh partial bytes on every exit, including cancellation and failures.
        if let Some(info) = self.available_models.lock().unwrap().get_mut(recipe.id) {
            info.is_downloaded = matches!(&result, Ok(true));
            info.is_downloading = false;
            self.bundle_states[index].invalidate_snapshot();
            info.partial_size = if info.is_downloaded {
                0
            } else {
                self.bundle_partial_size(recipe.id)
            };
        }
        if result? {
            let _ = self.app_handle.emit("model-download-complete", recipe.id);
        }
        Ok(())
    }

    pub(super) fn cancel_bundle_requests(&self, id: &str) {
        if let Ok((index, _)) = bundle_recipe(id) {
            self.bundle_states[index]
                .cancellation_epoch
                .fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(super) fn invalidate_bundle_status(&self, id: &str) {
        if let Ok((index, _)) = bundle_recipe(id) {
            self.bundle_states[index].invalidate_snapshot();
        }
    }

    pub(super) fn delete_bundle_model(&self, model: &ModelInfo) -> Result<()> {
        let (index, recipe) = bundle_recipe(&model.id)?;
        let _download = self.bundle_states[index]
            .download
            .try_lock()
            .context("Bundle download is still active; cancel it before deleting")?;
        for name in [
            recipe.id.to_string(),
            format!("{}.partial", recipe.id),
            format!("{}.extracting", recipe.id),
        ] {
            remove_path(&self.models_dir.join(name))?;
        }
        *self.bundle_states[index].verified.lock() = None;
        {
            let mut models = self
                .available_models
                .lock()
                .map_err(|_| anyhow::anyhow!("Model registry lock is poisoned"))?;
            if let Some(info) = models.get_mut(recipe.id) {
                info.is_downloaded = false;
                info.is_downloading = false;
                info.partial_size = 0;
                self.bundle_states[index].invalidate_snapshot();
            }
        }
        self.update_download_status()?;
        let _ = self.app_handle.emit("model-deleted", recipe.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
