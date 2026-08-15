//! PSV 教師 score の同点保持 quantile 較正ツール。
//!
//! `fit` は source/reference の train PSV から単調写像を作り、`apply` は保存済み写像を
//! PSV の score だけへ適用する。局面など残り 38 bytes は変更しない。

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

const RECORD_SIZE: usize = 40;
const SCORE_OFFSET: usize = 32;
const SCORE_VALUES: usize = 1 << 16;
const MAP_SCHEMA_VERSION: u32 = 1;
const MAP_METHOD: &str = "tie_preserving_quantile";
const ROUNDING: &str = "round_ties_even";
const UNSEEN_SCORE_RULE: &str = "linear_interpolation_with_endpoint_clamp";

#[derive(Parser, Debug)]
#[command(name = "calibrate_psv_scores")]
#[command(about = "PSV教師scoreをtrain-onlyの同点保持quantile写像で較正する")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// source/reference の train PSV から較正写像を作る
    Fit {
        /// 較正元のtrain PSV。複数ファイルは空白区切りで指定
        #[arg(long, required = true, num_args = 1..)]
        source_train: Vec<PathBuf>,

        /// 目標分布のtrain PSV。複数ファイルは空白区切りで指定
        #[arg(long, required = true, num_args = 1..)]
        reference_train: Vec<PathBuf>,

        /// 出力する較正map JSON（既存ファイルは上書きしない）
        #[arg(long)]
        output_map: PathBuf,
    },

    /// 保存済み写像をPSVへ適用する
    Apply {
        /// 入力PSV。複数ファイルは空白区切りで指定
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,

        /// `fit` で作成した較正map JSON
        #[arg(long)]
        map: PathBuf,

        /// 出力ディレクトリ（入力と同名で作成、既存ファイルは上書きしない）
        #[arg(long)]
        output_dir: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FileFingerprint {
    path: String,
    bytes: u64,
    records: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DatasetFingerprint {
    files: Vec<FileFingerprint>,
    records: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CalibrationMap {
    schema_version: u32,
    method: String,
    rounding: String,
    unseen_score_rule: String,
    source_train: DatasetFingerprint,
    reference_train: DatasetFingerprint,
    /// `[source_score, calibrated_score]`。source_score は厳密な昇順。
    mapping: Vec<[i16; 2]>,
}

#[derive(Debug)]
struct HistogramResult {
    histogram: Vec<u64>,
    fingerprint: DatasetFingerprint,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Fit {
            source_train,
            reference_train,
            output_map,
        } => fit(&source_train, &reference_train, &output_map),
        Command::Apply {
            input,
            map,
            output_dir,
        } => apply(&input, &map, &output_dir),
    }
}

fn fit(source_paths: &[PathBuf], reference_paths: &[PathBuf], output_map: &Path) -> Result<()> {
    ensure_output_absent(output_map, "output map")?;

    eprintln!("source trainを集計中...");
    let source = read_histogram(source_paths).context("source trainの集計に失敗")?;
    eprintln!("reference trainを集計中...");
    let reference = read_histogram(reference_paths).context("reference trainの集計に失敗")?;

    ensure!(source.fingerprint.records > 0, "train PSVが空です");
    ensure!(
        source.fingerprint.records == reference.fingerprint.records,
        "source/referenceのレコード数が一致しません: source={}, reference={}",
        source.fingerprint.records,
        reference.fingerprint.records
    );

    let mapping = fit_mapping(&source.histogram, &reference.histogram)?;
    let map = CalibrationMap {
        schema_version: MAP_SCHEMA_VERSION,
        method: MAP_METHOD.to_owned(),
        rounding: ROUNDING.to_owned(),
        unseen_score_rule: UNSEEN_SCORE_RULE.to_owned(),
        source_train: source.fingerprint,
        reference_train: reference.fingerprint,
        mapping,
    };
    validate_map(&map)?;
    write_json_atomic_noclobber(output_map, &map)?;

    eprintln!("写像数: {}", map.mapping.len());
    eprintln!("レコード数: {}", map.source_train.records);
    eprintln!("map出力: {}", output_map.display());
    Ok(())
}

fn apply(input_paths: &[PathBuf], map_path: &Path, output_dir: &Path) -> Result<()> {
    let (map, map_sha256) = load_map(map_path)?;
    let lookup = build_lookup_table(&map.mapping)?;
    let inputs = canonical_sorted_unique_paths(input_paths)?;

    fs::create_dir_all(output_dir)
        .with_context(|| format!("出力ディレクトリを作成できません: {}", output_dir.display()))?;
    let canonical_output_dir = output_dir
        .canonicalize()
        .with_context(|| format!("出力ディレクトリを解決できません: {}", output_dir.display()))?;
    ensure!(
        fs::metadata(&canonical_output_dir)?.is_dir(),
        "出力先はディレクトリではありません: {}",
        canonical_output_dir.display()
    );

    let outputs = preflight_apply_paths(&inputs, &canonical_output_dir)?;
    eprintln!("map SHA-256: {map_sha256}");
    eprintln!("入力ファイル数: {}", inputs.len());

    let mut total_records = 0u64;
    let mut total_changed = 0u64;
    for (input, output) in inputs.iter().zip(outputs.iter()) {
        let (records, changed, sha256) = apply_one(input, output, &lookup)?;
        total_records += records;
        total_changed += changed;
        eprintln!(
            "{}: {} records, {} changed, SHA-256 {}",
            output.display(),
            records,
            changed,
            sha256
        );
    }
    eprintln!("完了: {total_records} records, {total_changed} changed");
    Ok(())
}

fn read_histogram(paths: &[PathBuf]) -> Result<HistogramResult> {
    let paths = canonical_sorted_unique_paths(paths)?;
    let mut histogram = vec![0u64; SCORE_VALUES];
    let mut files = Vec::with_capacity(paths.len());
    let mut total_records = 0u64;

    for path in paths {
        let metadata = fs::metadata(&path)
            .with_context(|| format!("入力ファイルをstatできません: {}", path.display()))?;
        ensure!(metadata.is_file(), "入力は通常ファイルではありません: {}", path.display());
        ensure!(
            metadata.len().is_multiple_of(RECORD_SIZE as u64),
            "PSVサイズが{RECORD_SIZE}の倍数ではありません: {} ({} bytes)",
            path.display(),
            metadata.len()
        );
        let records = metadata.len() / RECORD_SIZE as u64;
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(&path)?);
        let mut record = [0u8; RECORD_SIZE];
        let mut hasher = Sha256::new();
        for _ in 0..records {
            reader.read_exact(&mut record)?;
            hasher.update(record);
            let score = i16::from_le_bytes([record[SCORE_OFFSET], record[SCORE_OFFSET + 1]]);
            histogram[score_index(score)] += 1;
        }
        let mut trailing = [0u8; 1];
        ensure!(
            reader.read(&mut trailing)? == 0,
            "読み取り中に入力ファイルのサイズが変わりました: {}",
            path.display()
        );
        total_records =
            total_records.checked_add(records).context("レコード数がu64を超えました")?;
        files.push(FileFingerprint {
            path: path.to_string_lossy().into_owned(),
            bytes: metadata.len(),
            records,
            sha256: hex_upper(&hasher.finalize()),
        });
    }

    Ok(HistogramResult {
        histogram,
        fingerprint: DatasetFingerprint {
            files,
            records: total_records,
        },
    })
}

fn fit_mapping(source: &[u64], reference: &[u64]) -> Result<Vec<[i16; 2]>> {
    ensure!(
        source.len() == SCORE_VALUES && reference.len() == SCORE_VALUES,
        "内部エラー: histogramサイズが不正です"
    );
    let source_total = histogram_total(source)?;
    let reference_total = histogram_total(reference)?;
    ensure!(source_total > 0, "source histogramが空です");
    ensure!(
        source_total == reference_total,
        "source/reference histogramの件数が一致しません"
    );

    let (reference_prefix_count, reference_prefix_sum) = prefix_histogram(reference)?;
    let mut mapping = Vec::new();
    let mut lo = 0u64;
    for (idx, &count) in source.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let hi = lo.checked_add(count).context("rankがu64を超えました")?;
        let sum = sorted_prefix_sum(hi, &reference_prefix_count, &reference_prefix_sum)?
            - sorted_prefix_sum(lo, &reference_prefix_count, &reference_prefix_sum)?;
        let calibrated = round_ratio_ties_even(sum, count)?;
        mapping.push([score_from_index(idx), calibrated]);
        lo = hi;
    }
    ensure!(lo == source_total, "内部エラー: source rankの終端が不一致です");
    Ok(mapping)
}

fn prefix_histogram(histogram: &[u64]) -> Result<(Vec<u64>, Vec<i128>)> {
    let mut counts = Vec::with_capacity(SCORE_VALUES + 1);
    let mut sums = Vec::with_capacity(SCORE_VALUES + 1);
    counts.push(0u64);
    sums.push(0i128);
    for (idx, &count) in histogram.iter().enumerate() {
        counts.push(counts.last().copied().unwrap().checked_add(count).context("件数overflow")?);
        let weighted = i128::from(score_from_index(idx)) * i128::from(count);
        sums.push(sums.last().copied().unwrap() + weighted);
    }
    Ok((counts, sums))
}

/// 昇順score列の先頭`rank`件の合計をhistogram prefixから返す。
fn sorted_prefix_sum(rank: u64, prefix_count: &[u64], prefix_sum: &[i128]) -> Result<i128> {
    let total = *prefix_count.last().context("空のprefix count")?;
    ensure!(rank <= total, "rankがreference件数を超えています: {rank} > {total}");
    if rank == total {
        return Ok(*prefix_sum.last().context("空のprefix sum")?);
    }
    let boundary = prefix_count.partition_point(|&count| count <= rank) - 1;
    let inside = rank - prefix_count[boundary];
    Ok(prefix_sum[boundary] + i128::from(score_from_index(boundary)) * i128::from(inside))
}

/// 有理数`numerator / denominator`をnearest, ties-to-evenで丸める。
fn round_ratio_ties_even(numerator: i128, denominator: u64) -> Result<i16> {
    ensure!(denominator > 0, "丸めの分母が0です");
    let negative = numerator < 0;
    let magnitude = numerator.abs();
    let denominator = i128::from(denominator);
    let quotient = magnitude / denominator;
    let remainder = magnitude % denominator;
    let rounded = match (remainder * 2).cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal => quotient + (quotient & 1),
    };
    let signed = if negative { -rounded } else { rounded };
    i16::try_from(signed).context("較正scoreがi16範囲外です")
}

fn build_lookup_table(mapping: &[[i16; 2]]) -> Result<Vec<i16>> {
    validate_mapping(mapping)?;
    let mut lookup = vec![0i16; SCORE_VALUES];
    let first = mapping[0];
    let last = *mapping.last().unwrap();
    let mut segment = 0usize;

    for (idx, slot) in lookup.iter_mut().enumerate() {
        let score = score_from_index(idx);
        *slot = if score <= first[0] {
            first[1]
        } else if score >= last[0] {
            last[1]
        } else {
            while mapping[segment + 1][0] < score {
                segment += 1;
            }
            let [x0, y0] = mapping[segment];
            let [x1, y1] = mapping[segment + 1];
            if score == x0 {
                y0
            } else if score == x1 {
                y1
            } else {
                let width = i128::from(i32::from(x1) - i32::from(x0));
                let offset = i128::from(i32::from(score) - i32::from(x0));
                let delta = i128::from(i32::from(y1) - i32::from(y0));
                let numerator = i128::from(y0) * width + offset * delta;
                round_ratio_ties_even(numerator, u64::try_from(width).unwrap())?
            }
        };
    }
    Ok(lookup)
}

fn load_map(path: &Path) -> Result<(CalibrationMap, String)> {
    let bytes = fs::read(path).with_context(|| format!("mapを読めません: {}", path.display()))?;
    let sha256 = hex_upper(&Sha256::digest(&bytes));
    let map: CalibrationMap = serde_json::from_slice(&bytes)
        .with_context(|| format!("map JSONをparseできません: {}", path.display()))?;
    validate_map(&map)?;
    Ok((map, sha256))
}

fn validate_map(map: &CalibrationMap) -> Result<()> {
    ensure!(
        map.schema_version == MAP_SCHEMA_VERSION,
        "未対応のmap schema_versionです: {} (expected {})",
        map.schema_version,
        MAP_SCHEMA_VERSION
    );
    ensure!(map.method == MAP_METHOD, "未対応のmap methodです: {}", map.method);
    ensure!(map.rounding == ROUNDING, "未対応の丸め規則です: {}", map.rounding);
    ensure!(
        map.unseen_score_rule == UNSEEN_SCORE_RULE,
        "未対応の未知score規則です: {}",
        map.unseen_score_rule
    );
    validate_mapping(&map.mapping)
}

fn validate_mapping(mapping: &[[i16; 2]]) -> Result<()> {
    ensure!(!mapping.is_empty(), "mapのmappingが空です");
    for pair in mapping.windows(2) {
        ensure!(
            pair[0][0] < pair[1][0],
            "mapのsource scoreが厳密な昇順ではありません: {} -> {}",
            pair[0][0],
            pair[1][0]
        );
        ensure!(
            pair[0][1] <= pair[1][1],
            "mapのcalibrated scoreが単調非減少ではありません: {} -> {}",
            pair[0][1],
            pair[1][1]
        );
    }
    Ok(())
}

fn preflight_apply_paths(inputs: &[PathBuf], output_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut names = HashSet::new();
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let metadata = fs::metadata(input)
            .with_context(|| format!("入力ファイルをstatできません: {}", input.display()))?;
        ensure!(metadata.is_file(), "入力は通常ファイルではありません: {}", input.display());
        ensure!(
            metadata.len().is_multiple_of(RECORD_SIZE as u64),
            "PSVサイズが{RECORD_SIZE}の倍数ではありません: {} ({} bytes)",
            input.display(),
            metadata.len()
        );
        let name = input.file_name().context("入力ファイル名を取得できません")?.to_owned();
        ensure!(
            names.insert(name.clone()),
            "複数入力のファイル名が衝突します: {}",
            name.to_string_lossy()
        );
        let output = output_dir.join(name);
        ensure!(input != &output, "入力と出力が同じファイルです: {}", input.display());
        ensure_output_absent(&output, "output PSV")?;
        outputs.push(output);
    }
    Ok(outputs)
}

fn apply_one(input: &Path, output: &Path, lookup: &[i16]) -> Result<(u64, u64, String)> {
    let bytes = fs::metadata(input)?.len();
    let records = bytes / RECORD_SIZE as u64;
    let parent = output.parent().context("出力先に親ディレクトリがありません")?;
    let mut temp = NamedTempFile::new_in(parent)
        .with_context(|| format!("一時出力を作成できません: {}", parent.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(input)?);
    let mut writer = BufWriter::with_capacity(1024 * 1024, temp.as_file_mut());
    let mut record = [0u8; RECORD_SIZE];
    let mut changed = 0u64;
    let mut hasher = Sha256::new();

    for _ in 0..records {
        reader.read_exact(&mut record)?;
        let old_score = i16::from_le_bytes([record[SCORE_OFFSET], record[SCORE_OFFSET + 1]]);
        let new_score = lookup[score_index(old_score)];
        if old_score != new_score {
            changed += 1;
        }
        record[SCORE_OFFSET..SCORE_OFFSET + 2].copy_from_slice(&new_score.to_le_bytes());
        writer.write_all(&record)?;
        hasher.update(record);
    }
    let mut trailing = [0u8; 1];
    ensure!(
        reader.read(&mut trailing)? == 0,
        "読み取り中に入力ファイルのサイズが変わりました: {}",
        input.display()
    );
    writer.flush()?;
    drop(writer);
    temp.as_file_mut().sync_all()?;
    temp.persist_noclobber(output)
        .map_err(|e| e.error)
        .with_context(|| format!("出力を確定できません: {}", output.display()))?;
    Ok((records, changed, hex_upper(&hasher.finalize())))
}

fn write_json_atomic_noclobber(path: &Path, value: &CalibrationMap) -> Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    ensure!(parent.exists(), "map出力の親ディレクトリが存在しません: {}", parent.display());
    let mut temp = NamedTempFile::new_in(parent)
        .with_context(|| format!("map一時ファイルを作成できません: {}", parent.display()))?;
    serde_json::to_writer_pretty(temp.as_file_mut(), value)?;
    temp.as_file_mut().write_all(b"\n")?;
    temp.as_file_mut().sync_all()?;
    temp.persist_noclobber(path)
        .map_err(|e| e.error)
        .with_context(|| format!("map出力を確定できません: {}", path.display()))?;
    Ok(())
}

fn canonical_sorted_unique_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    ensure!(!paths.is_empty(), "入力ファイルがありません");
    let mut result = paths
        .iter()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("入力パスを解決できません: {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    result.sort();
    for pair in result.windows(2) {
        ensure!(pair[0] != pair[1], "同じ入力ファイルが重複しています: {}", pair[0].display());
    }
    Ok(result)
}

fn ensure_output_absent(path: &Path, kind: &str) -> Result<()> {
    if fs::symlink_metadata(path).is_ok() {
        bail!("{kind}が既に存在します（上書きしません）: {}", path.display());
    }
    Ok(())
}

fn histogram_total(histogram: &[u64]) -> Result<u64> {
    histogram.iter().try_fold(0u64, |sum, &count| {
        sum.checked_add(count).context("histogram件数がu64を超えました")
    })
}

fn score_index(score: i16) -> usize {
    usize::try_from(i32::from(score) - i32::from(i16::MIN)).unwrap()
}

fn score_from_index(index: usize) -> i16 {
    debug_assert!(index < SCORE_VALUES);
    (i32::try_from(index).unwrap() + i32::from(i16::MIN)) as i16
}

fn hex_upper(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02X}").unwrap();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ratio_uses_ties_to_even_for_both_signs() {
        assert_eq!(round_ratio_ties_even(5, 2).unwrap(), 2);
        assert_eq!(round_ratio_ties_even(7, 2).unwrap(), 4);
        assert_eq!(round_ratio_ties_even(-5, 2).unwrap(), -2);
        assert_eq!(round_ratio_ties_even(-7, 2).unwrap(), -4);
        assert_eq!(round_ratio_ties_even(8, 3).unwrap(), 3);
        assert_eq!(round_ratio_ties_even(-8, 3).unwrap(), -3);
    }

    #[test]
    fn fit_mapping_preserves_source_ties_and_uses_reference_rank_interval() {
        let mut source = vec![0u64; SCORE_VALUES];
        let mut reference = vec![0u64; SCORE_VALUES];
        source[score_index(-10)] = 2;
        source[score_index(20)] = 1;
        source[score_index(30)] = 1;
        reference[score_index(-4)] = 1;
        reference[score_index(-1)] = 1;
        reference[score_index(8)] = 1;
        reference[score_index(12)] = 1;

        assert_eq!(fit_mapping(&source, &reference).unwrap(), vec![[-10, -2], [20, 8], [30, 12]]);
    }

    #[test]
    fn lookup_interpolates_unseen_scores_and_clamps_endpoints() {
        let lookup = build_lookup_table(&[[-10, -20], [0, 0], [4, 6], [10, 20]]).unwrap();
        assert_eq!(lookup[score_index(i16::MIN)], -20);
        assert_eq!(lookup[score_index(-5)], -10);
        assert_eq!(lookup[score_index(0)], 0);
        assert_eq!(lookup[score_index(2)], 3);
        assert_eq!(lookup[score_index(7)], 13);
        assert_eq!(lookup[score_index(i16::MAX)], 20);
    }

    #[test]
    fn lookup_interpolation_uses_ties_to_even() {
        let lookup = build_lookup_table(&[[0, 0], [4, 10]]).unwrap();
        assert_eq!(lookup[score_index(1)], 2); // 2.5 -> 2
        assert_eq!(lookup[score_index(3)], 8); // 7.5 -> 8
    }

    #[test]
    fn rejects_non_monotone_mapping() {
        assert!(build_lookup_table(&[[0, 0], [1, -1]]).is_err());
        assert!(build_lookup_table(&[[0, 0], [0, 1]]).is_err());
    }

    #[test]
    fn apply_changes_only_score_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.psv");
        let output = dir.path().join("output.psv");
        let mut bytes = [0u8; RECORD_SIZE * 2];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        bytes[SCORE_OFFSET..SCORE_OFFSET + 2].copy_from_slice(&(-10i16).to_le_bytes());
        let second = RECORD_SIZE + SCORE_OFFSET;
        bytes[second..second + 2].copy_from_slice(&(10i16).to_le_bytes());
        fs::write(&input, bytes).unwrap();

        let lookup = build_lookup_table(&[[-10, -3], [10, 7]]).unwrap();
        let (records, changed, _) = apply_one(&input, &output, &lookup).unwrap();
        assert_eq!(records, 2);
        assert_eq!(changed, 2);

        let actual = fs::read(output).unwrap();
        let mut expected = bytes;
        expected[SCORE_OFFSET..SCORE_OFFSET + 2].copy_from_slice(&(-3i16).to_le_bytes());
        expected[second..second + 2].copy_from_slice(&(7i16).to_le_bytes());
        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_refuses_existing_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.psv");
        let output_dir = dir.path().join("out");
        fs::create_dir(&output_dir).unwrap();
        fs::write(&input, [0u8; RECORD_SIZE]).unwrap();
        fs::write(output_dir.join("input.psv"), b"keep").unwrap();

        let inputs = canonical_sorted_unique_paths(&[input]).unwrap();
        let err = preflight_apply_paths(&inputs, &output_dir.canonicalize().unwrap()).unwrap_err();
        assert!(err.to_string().contains("上書きしません"));
        assert_eq!(fs::read(output_dir.join("input.psv")).unwrap(), b"keep");
    }
}
