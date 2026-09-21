use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn watched(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create native runtime staging directory");
    for entry in fs::read_dir(source).expect("read native runtime override") {
        let entry = entry.expect("read native runtime file");
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("stage native runtime file");
        }
    }
}

fn sherpa_source(out: &Path) -> PathBuf {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("native/dependencies.json"))
            .expect("parse pinned native dependency manifest");
    let pin = &manifest["sherpa"];
    let revision = pin["revision"].as_str().unwrap();
    let hash = pin["sha256"].as_str().unwrap();
    let extracted = out.join("sherpa-source");
    let source = extracted.join(format!("sherpa-onnx-{revision}"));
    let marker = extracted.join(".complete");
    if fs::read_to_string(&marker).ok().as_deref() == Some(hash) {
        return source;
    }
    let archive = out.join("sherpa-source.tar.gz");
    let status = Command::new(env::var_os("CMAKE").unwrap_or_else(|| "cmake".into()))
        .arg(format!("-DURL={}", pin["url"].as_str().unwrap()))
        .arg(format!("-DEXPECTED_SHA256={hash}"))
        .arg(format!("-DDESTINATION={}", archive.display()))
        .args(["-P", "native/FetchSource.cmake"])
        .status()
        .expect("run CMake to download pinned sherpa source");
    assert!(
        status.success(),
        "could not fetch verified sherpa source archive"
    );
    fs::create_dir_all(&extracted).expect("create native source extraction directory");
    let input = flate2::read::GzDecoder::new(fs::File::open(&archive).unwrap());
    let mut archive = tar::Archive::new(input);
    for entry in archive.entries().expect("read sherpa source archive") {
        let mut entry = entry.expect("read sherpa source archive entry");
        // Upstream contains unrelated absolute symlinks under scripts/go.
        // Never create symlinks/hardlinks/devices or follow archive link targets.
        if entry.header().entry_type().is_file() {
            assert!(
                entry
                    .unpack_in(&extracted)
                    .expect("safely unpack sherpa source"),
                "sherpa source archive attempted a path traversal"
            );
        }
    }
    assert!(
        source.join("CMakeLists.txt").is_file(),
        "unexpected sherpa archive root"
    );
    fs::write(marker, hash).expect("mark verified native source extraction");
    source
}

fn main() {
    println!("cargo:rerun-if-changed=native");
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    assert!(
        matches!(os.as_str(), "macos" | "linux" | "windows"),
        "X-ASR supports desktop macOS, Linux and Windows only"
    );
    assert!(
        matches!(arch.as_str(), "aarch64" | "x86_64"),
        "X-ASR requires an arm64 or x64 desktop target"
    );
    assert!(
        os != "windows" || target.ends_with("-msvc"),
        "X-ASR Windows builds require the MSVC ABI"
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let runtime = out.join("runtime");
    let prebuilt = watched("HANDY_XASR_NATIVE_DIR");
    let source = watched("HANDY_XASR_SOURCE_DIR");
    let dependencies = watched("HANDY_XASR_DEPENDENCY_DIR");
    let cmake_args = watched("HANDY_XASR_CMAKE_ARGS");
    let ort = watched("ORT_LIB_LOCATION");
    let includes = watched("ORT_INCLUDE_DIR");
    let notices = watched("ORT_LICENSE_DIR");
    watched("MACOSX_DEPLOYMENT_TARGET");
    watched("CMAKE");

    if let Some(directory) = prebuilt {
        // This is explicitly an override for a previously controlled build,
        // never an automatic download of stock sherpa binaries.
        println!("cargo:rerun-if-changed={directory}");
        copy_tree(Path::new(&directory), &runtime);
    } else {
        let source = source
            .map(PathBuf::from)
            .unwrap_or_else(|| sherpa_source(&out));
        let mut config = cmake::Config::new("native");
        config
            .profile("Release")
            .target(&target)
            .host(&host)
            .define("HANDY_XASR_RUNTIME_DIR", &runtime)
            .define("HANDY_XASR_TARGET_ARCH", &arch)
            .define("HANDY_XASR_SOURCE_DIR", &source)
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL")
            .build_target("handy_xasr_stage");
        match os.as_str() {
            "macos" => {
                config.define(
                    "CMAKE_OSX_ARCHITECTURES",
                    if arch == "aarch64" { "arm64" } else { "x86_64" },
                );
                let deployment_target = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| {
                    if arch == "aarch64" { "11.0" } else { "10.15" }.to_string()
                });
                config.define("CMAKE_OSX_DEPLOYMENT_TARGET", deployment_target);
            }
            "linux" => {
                config.define("CMAKE_SYSTEM_PROCESSOR", &arch);
            }
            "windows" => {
                config.define(
                    "CMAKE_SYSTEM_PROCESSOR",
                    if arch == "aarch64" { "ARM64" } else { "AMD64" },
                );
            }
            _ => unreachable!(),
        }
        for (name, value) in [
            ("HANDY_XASR_DEPENDENCY_DIR", dependencies),
            ("HANDY_ORT_ROOT", ort),
            ("HANDY_ORT_INCLUDE_DIR", includes),
            ("HANDY_ORT_LICENSE_DIR", notices),
        ] {
            if let Some(value) = value {
                config.define(name, value);
            }
        }
        if let Some(args) = cmake_args {
            for arg in shlex::split(&args)
                .expect("HANDY_XASR_CMAKE_ARGS must contain valid shell-quoted arguments")
            {
                config.configure_arg(arg);
            }
        }
        config.build();
    }

    let required: &[&str] = match os.as_str() {
        "macos" => &["libhandy_xasr.dylib", "libonnxruntime.dylib"],
        "linux" => &["libhandy_xasr.so", "libonnxruntime.so.1"],
        "windows" => &["handy_xasr.dll", "handy_xasr.lib", "onnxruntime.dll"],
        _ => unreachable!(),
    };
    for file in required {
        assert!(
            runtime.join(file).is_file(),
            "missing X-ASR runtime artifact: {}",
            runtime.join(file).display()
        );
    }
    assert!(
        runtime.join("licenses").is_dir(),
        "controlled X-ASR runtime must include third-party license notices"
    );
    println!("cargo:rustc-link-search=native={}", runtime.display());
    println!("cargo:rustc-link-lib=dylib=handy_xasr");
    // links=handy_xasr makes this DEP_HANDY_XASR_RUNTIME_DIR in Handy's build.rs.
    println!("cargo:runtime_dir={}", runtime.display());
    if matches!(os.as_str(), "macos" | "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", runtime.display());
    }
}
