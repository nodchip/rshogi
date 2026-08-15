# calibrate_psv_scores — PSV教師scoreの同点保持quantile較正

教師モデルのtrain score分布を、参照教師のtrain score分布へ単調に較正するツール。
source教師が付けた局面間の弱順序と同点を保ったまま、scoreの非線形CDFと端点飽和を
補正する。

処理は次の2段階に分かれる。

1. `fit`: source/referenceの**train PSVだけ**から較正mapを作る。
2. `apply`: 保存したmapをsource教師のtrain/test PSVへ適用する。

test PSVや対局結果を`fit`へ渡さないこと。写像をtrainだけで確定してから、同じ写像を
testや将来データへ適用する。

## ビルド

```bash
cargo build --release -p tools --bin calibrate_psv_scores
```

ONNX RuntimeやGPUは不要。

## mapを作る

```bash
cargo run --release -p tools --bin calibrate_psv_scores -- fit \
  --source-train data/aoba34/train.psv \
  --reference-train data/fuka_public/train.psv \
  --output-map data/calibration/aoba34-to-fuka-public.json
```

分割されたtrain PSVは、各オプションへ空白区切りで複数指定できる。

```bash
calibrate_psv_scores fit \
  --source-train source-000.psv source-001.psv \
  --reference-train reference-000.psv reference-001.psv \
  --output-map calibration.json
```

sourceとreferenceの総レコード数は一致している必要がある。入力ファイルはcanonical pathで
ソートしてから処理されるため、引数の順序を変えても同じ環境ではmapがbit一致する。

## mapを適用する

```bash
cargo run --release -p tools --bin calibrate_psv_scores -- apply \
  --input data/aoba34/train.psv data/aoba34/test.psv \
  --map data/calibration/aoba34-to-fuka-public.json \
  --output-dir data/aoba34-calibrated
```

出力ファイル名は入力と同じ。異なるディレクトリの同名入力を同時指定すると、出力衝突を
避けるため起動時に失敗する。既存の出力ファイルは上書きしない。

`apply`が変更するのは各40-byte PSV recordのscore 2 bytesだけで、局面、move、game ply、
game result、paddingを含む残り38 bytesはそのまま保持する。書き込みは同じ出力ディレクトリの
一時ファイルへ完了後、既存ファイルを上書きしないrenameで確定する。

## 変換の定義

source trainの各score同点群が、sourceを昇順に並べたときに占めるrank区間を
`[lo, hi)`とする。対応するreference sorted scoreの同じ区間の平均を、そのsource scoreの
変換先とする。

```text
mapped(source_score) = round_ties_even(mean(reference_sorted[lo..hi]))
```

- sourceで同じscoreを持つ局面は、変換後も必ず同じscoreになる。
- source score間の弱順序は変えない。
- referenceの局面別順位は使わず、score分布だけを参照する。
- trainに存在しないscoreは、隣接する既知scoreの写像間を線形補間して
  `round_ties_even`する。
- train範囲外のscoreは、写像の両端へclampする。

丸めはnearest, ties-to-even（ちょうど`.5`なら偶数側）である。Rustの`f64::round`
（ties away from zero）とは異なるため、有理数の整数演算で明示的に実装している。

## メモリと再現情報

`fit`はPSV全件をメモリへloadしない。`i16` scoreごとの65,536-bin `u64` histogramを
source/referenceに1個ずつ持ち、各入力をstreaming走査するため、ピークメモリは入力件数に
依存しない。

map JSONには次を記録する。

- schema version、方式名、丸め規則、未知score規則
- source/reference trainのファイル別path、bytes、records、SHA-256
- 総レコード数
- sourceの各既知scoreと変換後score

`apply`は読み込んだmap JSON自体のSHA-256と、各出力PSVのSHA-256をstderrへ表示する。

## 実験で確認した既知値

AobaZero WCSC34 train 12,000件をsource、ふかうら王公開train 12,000件をreferenceとして
fitした写像は4,212入力scoreを持つ。既存のtrain-only tie-preserving quantile実験と
次の出力がbit一致する。

| split | SHA-256 |
|---|---|
| train | `16AF84B2B182A9FD6DC8D793608AC81C8F725C20B9C8C42EFA479D9FD9A5EF52` |
| test | `D87C6D94670A4A4898EF7AE8A7394AE0B18B26FEFA829F33280A9EFA088DDEED` |
