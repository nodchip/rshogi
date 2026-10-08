#![cfg(windows)]

use std::fs;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ID: &str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[test]
fn terminal_meta_and_display_follow_observed_engine_exit() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tournament-exit-integration");
    fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let engine = dir.path().join("engine.exe");
    let compilation = Command::new("rustc")
        .arg(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tournament_exit_mock.rs"),
        )
        .arg("-o")
        .arg(&engine)
        .output()
        .unwrap();
    assert!(compilation.status.success(), "{}", String::from_utf8_lossy(&compilation.stderr));

    for mode in ["normal", "nonzero", "ignore", "startup_fail"] {
        let output = dir.path().join(mode);
        let display = dir.path().join(format!("{mode}-progress.txt"));
        let stderr = dir.path().join(format!("{mode}-stderr.txt"));
        let mut child = OwnedChild(
            Command::new(env!("CARGO_BIN_EXE_tournament"))
                .arg("--engine")
                .arg(&engine)
                .args(["--engine-label", "a", "--engine"])
                .arg(&engine)
                .args([
                    "--engine-label",
                    "b",
                    "--games",
                    "1",
                    "--nodes",
                    "1",
                    "--concurrency",
                    "1",
                    "--out-dir",
                ])
                .arg(&output)
                .arg("--progress-file")
                .arg(&display)
                .args(["--progress-experiment-id", ID, "--progress-run-id", ID])
                .env("TOURNAMENT_EXIT_MODE", mode)
                .stdout(Stdio::null())
                .stderr(fs::File::create(&stderr).unwrap())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        let status = loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "{mode}: {}", fs::read_to_string(&stderr).unwrap());
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(
            status.success(),
            mode == "normal",
            "{mode}: {}",
            fs::read_to_string(stderr).unwrap()
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["engine_exits"]["observations_complete"], true);
        assert_eq!(
            meta["run_status"],
            if mode == "normal" {
                "completed"
            } else {
                "worker_failed"
            }
        );
        assert_eq!(meta["invalid"], mode != "normal");
        if mode == "startup_fail" {
            assert_eq!(meta["engine_exits"]["unconfirmed"], 1);
            assert_eq!(meta["engine_exits"]["spawned"], 0);
        } else {
            assert_eq!(meta["engine_exits"]["spawned"], 2);
            assert_eq!(meta["engine_exits"]["reaped"], 2);
            assert_eq!(meta["engine_exits"]["unconfirmed"], 0);
            assert_eq!(meta["engine_exits"]["forced"], if mode == "ignore" { 2 } else { 0 });
            assert_eq!(meta["engine_exits"]["non_success"], if mode == "normal" { 0 } else { 2 });
            let rows = fs::read_to_string(output.join("pair-0-1__a-vs-b.jsonl")).unwrap();
            assert_eq!(
                rows.lines()
                    .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                    .filter(|row| row["type"] == "result")
                    .count(),
                2
            );
        }
        let text = fs::read_to_string(display).unwrap();
        assert!(text.contains("state: result_unchecked\n"));
        assert!(!text.contains("succeeded"));
        if mode != "startup_fail" {
            assert!(text.contains("received_games: 2\nvalid_games: 2\n"));
        }
    }
}
