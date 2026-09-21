use super::*;
use bzip2::write::BzEncoder;
use bzip2::Compression;
use tempfile::TempDir;

const FILES: &[BundleFile] = &[
    BundleFile {
        name: "encoder.onnx",
        size: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    },
    BundleFile {
        name: "tokens.txt",
        size: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    },
];

fn fixture_recipe() -> BundleRecipe {
    BundleRecipe {
        id: "fixture",
        name: "fixture",
        description: "fixture",
        files: FILES,
        engine: EngineType::XAsrOffline,
        archive: Some(BundleArchive {
            url: "",
            root: "matched",
            size: 0,
            sha256: "",
        }),
    }
}

fn archive_fixture(path: &Path, entries: &[(&str, &[u8])]) {
    let encoder = BzEncoder::new(File::create(path).unwrap(), Compression::fast());
    let mut archive = tar::Builder::new(encoder);
    for (name, content) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append_data(&mut header, name, *content).unwrap();
    }
    archive.into_inner().unwrap().finish().unwrap();
}

#[test]
fn incomplete_bundle_is_not_resolvable_until_every_component_matches() {
    let dir = TempDir::new().unwrap();
    assert!(verify_components(dir.path(), FILES).is_err());
    fs::write(dir.path().join("encoder.onnx"), b"abc").unwrap();
    assert!(verify_components(dir.path(), FILES).is_err());
    fs::write(dir.path().join("tokens.txt"), b"ab").unwrap();
    assert!(verify_components(dir.path(), FILES).is_err());
    fs::write(dir.path().join("tokens.txt"), b"abc").unwrap();
    assert!(verify_components(dir.path(), FILES).is_ok());
}

#[test]
fn same_size_mismatched_component_is_rejected_without_deleting_dropin() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("encoder.onnx"), b"abc").unwrap();
    let tokens = dir.path().join("tokens.txt");
    fs::write(&tokens, b"abd").unwrap();
    assert!(verify_components(dir.path(), FILES).is_err());
    assert_eq!(fs::read(&tokens).unwrap(), b"abd");
}

#[test]
fn archive_requires_complete_matched_root_and_component_hashes() {
    let cases: &[&[(&str, &[u8])]] = &[
        &[("matched/encoder.onnx", b"abc")],
        &[
            ("other/encoder.onnx", b"abc"),
            ("matched/tokens.txt", b"abc"),
        ],
        &[
            ("matched/encoder.onnx", b"abc"),
            ("matched/tokens.txt", b"abd"),
        ],
    ];
    for entries in cases {
        let dir = TempDir::new().unwrap();
        let archive = dir.path().join("bundle.tar.bz2");
        let output = dir.path().join("extracting");
        fs::create_dir(&output).unwrap();
        archive_fixture(&archive, entries);
        assert!(extract_archive(
            &archive,
            &output,
            &fixture_recipe(),
            &CancellationToken::new()
        )
        .is_err());
        assert!(verify_components(&output, FILES).is_err());
    }

    let dir = TempDir::new().unwrap();
    let archive = dir.path().join("bundle.tar.bz2");
    let output = dir.path().join("extracting");
    fs::create_dir(&output).unwrap();
    archive_fixture(
        &archive,
        &[
            ("matched/encoder.onnx", b"abc"),
            ("matched/tokens.txt", b"abc"),
        ],
    );
    assert!(extract_archive(
        &archive,
        &output,
        &fixture_recipe(),
        &CancellationToken::new()
    )
    .unwrap());
    assert_eq!(fs::read(output.join("encoder.onnx")).unwrap(), b"abc");
    assert_eq!(fs::read(output.join("tokens.txt")).unwrap(), b"abc");
}

#[test]
fn archive_does_not_resolve_components_via_links() {
    let dir = TempDir::new().unwrap();
    let archive_path = dir.path().join("bundle.tar.bz2");
    let encoder = BzEncoder::new(File::create(&archive_path).unwrap(), Compression::fast());
    let mut archive = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o644);
    archive
        .append_link(&mut header, "matched/encoder.onnx", "../../outside.onnx")
        .unwrap();
    archive.into_inner().unwrap().finish().unwrap();
    let output = dir.path().join("extracting");
    fs::create_dir(&output).unwrap();
    assert!(extract_archive(
        &archive_path,
        &output,
        &fixture_recipe(),
        &CancellationToken::new()
    )
    .is_err());
    assert!(!output.join("encoder.onnx").exists());
}

#[tokio::test]
async fn cancelling_a_queued_retry_does_not_restart_the_download() {
    use std::future::{poll_fn, Future};
    use std::task::Poll;

    let state = BundleState::default();
    let predecessor = state.download.lock().await;
    let mut retry = Box::pin(state.acquire_download());
    // Poll exactly once while the predecessor owns the directory, without
    // depending on task scheduling or a timing-sensitive sleep.
    let first_poll = poll_fn(|cx| Poll::Ready(retry.as_mut().poll(cx))).await;
    assert!(first_poll.is_pending());
    state.cancellation_epoch.fetch_add(1, Ordering::AcqRel);
    drop(predecessor);
    assert!(retry.await.is_none());
    assert!(state.acquire_download().await.is_some());
}

#[test]
fn a_slow_rescan_cannot_replace_completed_or_deleted_bundle_status() {
    let state = BundleState::default();
    let mut models = HashMap::new();
    register_models(&mut models);
    let model = models.get_mut(RECIPES[0].id).unwrap();
    let before_completion = state.revision.load(Ordering::Acquire);
    let incomplete = DiskStatus {
        partial_size: 123,
        ..DiskStatus::default()
    };
    model.is_downloaded = true;
    model.is_downloading = false;
    state.invalidate_snapshot();
    state.apply_snapshot(model, before_completion, &incomplete, true);
    assert!(model.is_downloaded);
    assert!(!model.is_downloading);
    assert_eq!(model.partial_size, 0);

    let before_deletion = state.revision.load(Ordering::Acquire);
    let complete = DiskStatus {
        is_downloaded: true,
        ..DiskStatus::default()
    };
    model.is_downloaded = false;
    state.invalidate_snapshot();
    state.apply_snapshot(model, before_deletion, &complete, false);
    assert!(!model.is_downloaded);
}
