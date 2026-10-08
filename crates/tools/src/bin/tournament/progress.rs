//! 集計イベントから更新する、結果検査とは独立した進捗表示。

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use chrono::Utc;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
type PairStats = HashMap<(usize, usize), (u32, u32, u32)>;

pub(super) fn parse_id(value: &str) -> Result<String, String> {
    if value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
    {
        Ok(value.into())
    } else {
        Err("progress identity must use UUID notation".into())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Snapshot<'a> {
    pub received_games: u32,
    pub valid_games: u32,
    pub target_valid_games: u32,
    pub elapsed: Duration,
    pub collection_finished: bool,
    pub pair_stats: &'a PairStats,
}

pub(super) struct Display {
    path: PathBuf,
    experiment_id: String,
    run_id: String,
    last_write: Option<Instant>,
    warned: bool,
}

impl Display {
    pub(super) fn new(path: PathBuf, experiment_id: String, run_id: String) -> io::Result<Self> {
        if !path.is_absolute()
            || path.file_name().is_none()
            || parse_id(&experiment_id).is_err()
            || parse_id(&run_id).is_err()
        {
            return Err(io::Error::other(
                "progress file requires an absolute file path and UUID ids",
            ));
        }
        Ok(Self {
            path,
            experiment_id,
            run_id,
            last_write: None,
            warned: false,
        })
    }

    /// 明示的な入力と対局出力から表示先を分離し、engine起動前に拒否する。
    pub(super) fn validate_paths<'a>(
        &self,
        output: &Path,
        inputs: impl Iterator<Item = &'a Path>,
    ) -> io::Result<()> {
        let target = tools::common::dedup::canonicalize_maybe_new(&self.path)
            .map_err(|_| io::Error::other("progress path cannot be checked"))?;
        let output = tools::common::dedup::canonicalize_maybe_new(output)
            .map_err(|_| io::Error::other("tournament output path cannot be checked"))?;
        let inside_output = target.starts_with(&output);
        // 未作成部分の大文字小文字もWindowsでは同じ保存先になり得る。
        #[cfg(windows)]
        let inside_output = inside_output
            || Path::new(&target.to_string_lossy().to_lowercase())
                .starts_with(Path::new(&output.to_string_lossy().to_lowercase()));
        if inside_output {
            return Err(io::Error::other(
                "progress file must be outside the tournament output directory",
            ));
        }
        for input in inputs {
            tools::output_path::ensure_safe_output_path(&self.path, input).map_err(|_| {
                io::Error::other("progress file conflicts with input or cannot be checked")
            })?;
        }
        Ok(())
    }

    /// timerを持たず、実結果の取込み時に通常10秒で間引く。障害後も実イベントでだけ試す。
    pub(super) fn update(&mut self, snapshot: Snapshot<'_>, force: bool) {
        self.update_at(snapshot, force, Instant::now());
    }

    fn update_at(&mut self, snapshot: Snapshot<'_>, force: bool, now: Instant) {
        if !force
            && self
                .last_write
                .is_some_and(|last| now.duration_since(last) < Duration::from_secs(10))
        {
            return;
        }
        self.last_write = Some(now);
        if self.write_snapshot(snapshot).is_err() && !self.warned {
            self.warned = true;
            eprintln!("warning: progress_file_write_failed; run_id={}", self.run_id);
        }
    }

    fn write_snapshot(&self, snapshot: Snapshot<'_>) -> io::Result<()> {
        let parent =
            self.path.parent().ok_or_else(|| io::Error::other("progress parent missing"))?;
        let temp = parent.join(format!(
            ".rating-progress-{}-{}-{}.tmp",
            self.run_id,
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
        let result = (|| {
            writeln!(file, "updated_at: {}", Utc::now().to_rfc3339())?;
            writeln!(file, "experiment_id: {}", self.experiment_id)?;
            writeln!(file, "run_id: {}", self.run_id)?;
            writeln!(file, "operation: rating")?;
            writeln!(
                file,
                "state: {}",
                if snapshot.collection_finished {
                    "result_unchecked"
                } else {
                    "running"
                }
            )?;
            writeln!(file, "received_games: {}", snapshot.received_games)?;
            writeln!(file, "valid_games: {}", snapshot.valid_games)?;
            writeln!(file, "target_valid_games: {}", snapshot.target_valid_games)?;
            writeln!(file, "elapsed_seconds: {:.1}", snapshot.elapsed.as_secs_f64())?;
            for ((i, j), (wins, losses, draws)) in snapshot.pair_stats {
                writeln!(file, "pair_{i}_{j}: W={wins} D={draws} L={losses}")?;
            }
            writeln!(file, "completion_requires: process_exit_and_result_checks")?;
            file.flush()?;
            drop(file);
            fs::rename(&temp, &self.path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";

    fn directory() -> tempfile::TempDir {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/progress-tests");
        fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }

    fn snapshot(received_games: u32) -> Snapshot<'static> {
        static STATS: std::sync::OnceLock<PairStats> = std::sync::OnceLock::new();
        Snapshot {
            received_games,
            valid_games: received_games / 2 * 2,
            target_valid_games: 6,
            elapsed: Duration::from_secs(12),
            collection_finished: false,
            pair_stats: STATS.get_or_init(HashMap::new),
        }
    }

    #[test]
    fn snapshots_are_throttled_complete_and_never_claim_success() {
        let dir = directory();
        let path = dir.path().join("progress.txt");
        let mut display = Display::new(path.clone(), ID.into(), ID.into()).unwrap();
        let now = Instant::now();
        display.update_at(snapshot(1), true, now);
        let first = fs::read_to_string(&path).unwrap();
        assert!(first.contains("received_games: 1\nvalid_games: 0\n"));
        display.update_at(snapshot(2), false, now + Duration::from_secs(9));
        assert_eq!(fs::read_to_string(&path).unwrap(), first);
        display.update_at(snapshot(3), false, now + Duration::from_secs(10));
        assert!(
            fs::read_to_string(&path)
                .unwrap()
                .contains("received_games: 3\nvalid_games: 2\n")
        );
        let stats = [((0, 1), (2, 1, 3))].into();
        let mut last = Snapshot {
            pair_stats: &stats,
            ..snapshot(6)
        };
        last.collection_finished = true;
        display.update_at(last, true, now + Duration::from_secs(11));
        let text = fs::read_to_string(path).unwrap();
        assert!(text.contains("state: result_unchecked\n"));
        assert!(text.contains("pair_0_1: W=2 D=3 L=1\n"));
        assert!(!text.contains("succeeded"));
        assert!(!text.contains(dir.path().to_str().unwrap()));
    }

    #[test]
    fn failed_replace_preserves_old_display_and_recovers_on_later_event() {
        let dir = directory();
        let path = dir.path().join("progress.txt");
        fs::create_dir(&path).unwrap();
        fs::write(path.join("old.txt"), "old snapshot").unwrap();
        let mut display = Display::new(path.clone(), ID.into(), ID.into()).unwrap();
        let now = Instant::now();
        display.update_at(snapshot(1), true, now);
        assert!(display.warned);
        assert_eq!(fs::read_to_string(path.join("old.txt")).unwrap(), "old snapshot");
        fs::remove_file(path.join("old.txt")).unwrap();
        fs::remove_dir(&path).unwrap();
        display.update_at(snapshot(2), false, now + Duration::from_secs(9));
        assert!(!path.exists());
        display.update_at(snapshot(2), false, now + Duration::from_secs(10));
        assert!(fs::read_to_string(path).unwrap().contains("received_games: 2\n"));
    }

    #[test]
    fn identities_and_input_output_separation_are_required() {
        let dir = directory();
        let input = dir.path().join("opening.txt");
        fs::write(&input, "fixture").unwrap();
        let output = dir.path().join("results");
        let display = Display::new(input.clone(), ID.into(), ID.into()).unwrap();
        assert!(display.validate_paths(&output, std::iter::once(input.as_path())).is_err());
        #[cfg(windows)]
        {
            let display =
                Display::new(dir.path().join("RESULTS/meta.json"), ID.into(), ID.into()).unwrap();
            assert!(display.validate_paths(&output, std::iter::once(input.as_path())).is_err());
        }
        let display = Display::new(output.join("meta.json"), ID.into(), ID.into()).unwrap();
        assert!(display.validate_paths(&output, std::iter::once(input.as_path())).is_err());
        assert!(Display::new(PathBuf::from("relative.txt"), ID.into(), ID.into()).is_err());
        assert!(
            Display::new(dir.path().join("progress.txt"), "private\npath".into(), ID.into())
                .is_err()
        );
        assert_eq!(fs::read_to_string(input).unwrap(), "fixture");
    }

    #[cfg(windows)]
    #[test]
    fn locked_display_preserves_original_bytes_and_recovers_after_unlock() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = directory();
        let path = dir.path().join("progress.txt");
        fs::write(&path, "old snapshot").unwrap();
        let held = OpenOptions::new().read(true).share_mode(1).open(&path).unwrap();
        let mut display = Display::new(path.clone(), ID.into(), ID.into()).unwrap();
        display.update(snapshot(1), true);
        assert!(display.warned);
        assert_eq!(fs::read_to_string(&path).unwrap(), "old snapshot");
        drop(held);
        display.update(snapshot(2), true);
        assert!(fs::read_to_string(&path).unwrap().contains("received_games: 2\n"));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
