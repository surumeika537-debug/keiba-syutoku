# keiba-trifecta-analysis

JRA重賞（G1/G2/G3）の三連単期待値を分析するための **データ基盤 + バックテスト基盤**。
最初からAI予測モデルは作らず、まず **データ取得 → DB保存 → 整形 → 簡易バックテスト** までを動かす構成。

> ⚠️ スクレイピング元サイトの **利用規約 / robots.txt** は利用者自身で確認してください。
> デフォルトのアクセス間隔は3秒に設定済みです。短くしないでください。

---

## ディレクトリ構成

```
keiba-trifecta-analysis/
├─ data/
│  ├─ raw/race_results/   # 取得済みHTML（再取得しない）
│  ├─ processed/          # 整形済みCSV / バックテスト結果
│  └─ db/                 # SQLite (keiba.sqlite)
├─ notebooks/             # 分析用 Jupyter
├─ scripts/
│  ├─ init_db.py
│  ├─ ingest/             # 取得層 (base + 各ソース実装)
│  ├─ transform/          # 整形層 (base + 各ソース用パーサ)
│  └─ backtest/           # バックテスト
├─ src/
│  ├─ config.py           # パス・取得設定
│  ├─ database.py         # SQLAlchemy エンジン
│  ├─ schemas.py          # races / entries / payouts テーブル定義
│  └─ utils.py
├─ requirements.txt
└─ .gitignore
```

---

## セットアップ

### 1. 仮想環境を作る

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

bash:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 依存パッケージをインストール

```powershell
pip install -r requirements.txt
```

### 3. DBを初期化

```powershell
python scripts\init_db.py
```
→ `data/db/keiba.sqlite` が作られ、`races` / `entries` / `payouts` テーブルが作成される。

### 4. レース結果HTMLを取得

```powershell
# 直近3年分のG1/G2/G3
python scripts\ingest\fetch_race_results.py --years 2023 2024 2025
```

オプション:
- `--grades G1 G2 G3` 取得するグレード（デフォルト全部）
- `--race-ids-file ids.txt` 自動discoveryをスキップしてrace_id一覧から取得
- `--limit 20` デバッグ用に最初のN件だけ
- `--source netkeiba` 取得元（デフォルト netkeiba）

挙動:
- `data/raw/race_results/{race_id}.html` に保存
- **既に保存済みのファイルは再取得しない**
- リクエスト間に `FETCH_SLEEP_SECONDS`（デフォルト3秒）+ ジッタを入れる

### 5. パース → DB & CSV

```powershell
python scripts\transform\parse_race_results.py
```
→ `data/raw/race_results/*.html` を読み、`races` / `entries` / `payouts` に投入。
→ 同時に `data/processed/{races,entries,payouts}.csv` を書き出す。

再パースしたい場合:
```powershell
python scripts\transform\parse_race_results.py --rebuild
```

### 6. バックテスト

```powershell
python scripts\backtest\simple_roi.py
```

結果（標準出力 + `data/processed/backtest_results.csv`）の列:

`戦略 / 対象レース数 / 買い目点数 / 投資額(円) / 的中数 / 払戻額(円) / ROI / 的中率`

戦略:
- **A**: 1番人気を1着固定、2〜5番人気を2着・3着に流す
- **B**: 1〜3番人気の三連単ボックス
- **C**: 1番人気を1着固定、4〜8番人気を2着・3着に流す
- **D**: 1番人気 + 2番人気 + (5番人気以下 かつ 単勝オッズ10〜30倍) の馬を1頭含む三連単ボックス
- **E_D9_P3_CAP4** ⭐ : 本命運用ルール (詳細は次セクション)

フィルタ:
```powershell
python scripts\backtest\simple_roi.py --grades G1 --from 2024-01-01 --to 2024-12-31
```

---

## 本命運用ルール: E_D9_P3_CAP4

10年データの段階探索 (deepdive → variants → bankroll → pruning → cap比較) を経て決まった、
**現時点で唯一統計的にプラスROI が保証される運用ルール**。

### ルール定義

**買うレース (D9 plus候補) — 以下のいずれか:**
- **niigata_4g3**: 新潟 G3 のうち アイビスサマーD / 関屋記念 / 新潟記念 / レパードS
- **march_g2**: 3月 開催 の G2
- **hanshin_g1**: 阪神 開催 の G1
- **september_g3**: 9月 開催 の G3

**買わないレース (D6 negative filter):**
- 京都 G1
- 中山 G1
- 小倉 G3

**買い目 (P3 cap4):**
1. 1番人気 (p1) と 2番人気 (p2) を確定
2. dark horse 候補を抽出: `popularity >= 5` かつ `win_odds ∈ [10, 30]`
3. 17-18 頭立てなら、dark horse の 3-4 枠は除外 (D6 subset filter)
4. dark を **win_odds 昇順 (同値時 popularity 昇順、同値時 horse_number 昇順) で 上位 4 頭まで**
5. 各 dark について 2 点購入 (P3 = dark を 3 着固定、p1/p2 を 1着2着で swap):
   - `p1 → p2 → dark`
   - `p2 → p1 → dark`
6. 1点 100円固定、最大 8 点 / レース (最小 2 点)

### 期待値 (10年 backtest)

| 指標 | 値 |
|---|---:|
| 対象レース数 | 187 / 1,289 (= 約 19/年) |
| ROI | **+141.0%** |
| Bootstrap 95% CI | **[+6.2%, +293.6%]** (両端プラス = 統計的有意) |
| 的中数 | 15 / 187 (= 8.0%) |
| Avg ticket数 | 6.0 / race |
| Avg コスト | 600 円 / race |
| MC 破産率 (100K初期) | **0.00%** |
| MC p95 max DD | 26.2% |
| worst_first シナリオ | +159% で生存 (DD 30%) |

### 推奨初期資金と運用

| 初期資金 | bet/ticket | 1レース上限 | 期待 final | 推奨度 |
|---:|---:|---:|---:|---|
| **100,000円** | **100円** | **800円** | 258,720円 (+159%) | ⭐⭐⭐ |
| 50,000円 | 100円 | 800円 | 208,720円 (+317%) | ⭐ (余裕は減る) |
| 300,000円 | 100円 | 800円 | 458,720円 (+53%) | ⭐⭐ (安全度↑) |

### 週末 paper trading の流れ

E_D9_P3_CAP4 を実投票せずに「発走前 odds で買い目を立て、レース後に答え合わせ」する
公式ワークフロー。**snapshot (発走前) と final (確定 odds) の両方を生成し、ズレを確認
することで、リアルタイム運用での再現性を測る**。

```powershell
# === 前日〜直前 ===
# 1. データ更新 (週末の対象レース取得 + パース)
python scripts\ingest\fetch_race_results.py --years 2026
python scripts\transform\parse_race_results.py --rebuild

# 2a. 発走 30 分前 snapshot で買い目生成 (※現状は DB が確定 odds のみのため
#     placeholder としてラベルだけ "30min" に切り替えて記録される)
python scripts\live\generate_tickets.py --date 2026-09-06 --snapshot-time 30min
# → data\processed\live_tickets_2026-09-06_30min.csv

# === 全レース終了後 ===
# 2b. 確定 odds で再生成 (snapshot_time=final)
python scripts\live\generate_tickets.py --date 2026-09-06 --snapshot-time final
# → data\processed\live_tickets_2026-09-06_final.csv

# 3. 発走前 vs 確定 odds の差分確認 (ticket / dark horse / p1p2 のズレを見る)
python scripts\live\compare_snapshot_vs_final.py `
    --snapshot data\processed\live_tickets_2026-09-06_30min.csv `
    --final    data\processed\live_tickets_2026-09-06_final.csv
# → data\processed\snapshot_vs_final_diff.csv

# 4. 実際の払戻と突き合わせて記録 (snapshot 版で実投票したつもりで記録するのが本流)
python scripts\live\record_result.py `
    --tickets data\processed\live_tickets_2026-09-06_30min.csv `
    --bankroll 100000
# → data\processed\paper_trading_log.csv (1行 = 1レース)

# 5. 累計成績の確認: paper_trading_log.csv を Excel/Jupyter で開いて
#    cumulative_profit_yen / bankroll_after / hit_flag の推移を可視化
```

**paper_trading_log.csv の主要列** (per race):
- メタ: `rule_name / snapshot_time / odds_source / generated_at / final_result_checked_at`
- 買い目: `total_tickets / total_stake_yen / tickets_bought` (`;`-join)
- 結果: `hit_flag / payout_yen / profit_yen`
- 累計: `cumulative_profit_yen / bankroll_after`
- 備考: `notes` (hit / miss と実際の三連単 combo)

**snapshot_vs_final_diff.csv の主要列** (per race):
- `eligibility_status`: both / snapshot_only / final_only (E ルール対象になったか)
- `n_tickets_snapshot / n_tickets_final / common_count / *_only_count`
- `p1_changed / p2_changed / p1_odds_snapshot / p1_odds_final`
- `p1_pop_drift_for_snapshot_horse` (snapshot 時 1番人気だった馬の final 順位 - 1)
- `p2_pop_drift_for_snapshot_horse`
- `darks_added / darks_removed` (発走前→確定で出入りした dark horse 番号 `;`-join)
- `darks_reordered` (集合は同じだが順位が変わった)
- `actual_trifecta_combo / snapshot_would_hit / final_would_hit / ticket_hit_drift / hit_direction`
  (`hit_direction` ∈ both_hit / both_miss / gained_in_snapshot / lost_in_snapshot / no_result_in_db)

---

## 空 rebuild 防止 (parse_race_results.py の safety guards)

`scripts/transform/parse_race_results.py --rebuild` は **DB 全消し → 再投入** という
危険な操作。過去に「fetch が新規 race を取れず、rebuild が空のまま走って既存 1289
race を消す寸前」というインシデントがあったため、以下 3 段の安全装置を組込み済:

### 1. 空 raw dir → ABORT

```bash
# raw HTML が 0 件のディレクトリで --rebuild は exit 2 で停止
python scripts/transform/parse_race_results.py --rebuild --raw-dir /tmp/empty
# → ERROR: ABORT: --rebuild with zero raw HTML files would wipe the DB.
# →        If this is really what you want, pass --allow-empty-rebuild.
```

### 2. 低 races threshold → ABORT

```bash
# parse 結果の race 数が --min-races-threshold (default 100) 未満なら exit 3
python scripts/transform/parse_race_results.py --rebuild --min-races-threshold 9999
# → ERROR: ABORT: parsed_races=1878 < min_threshold=9999.
# →        Refusing to touch DB to avoid wiping good data.
```

初回少量取込時は `--min-races-threshold 0` で明示的に無効化可。

### 3. FK-safe rebuild (odds_snapshots を保持)

`odds_snapshots.race_id` は `races.race_id` に FK 参照。普通に `DELETE FROM races`
すると FK 違反で IntegrityError → rollback (= rebuild が常に失敗) という症状になる。

修正版は `PRAGMA defer_foreign_keys = ON` でトランザクション末まで FK チェック
を遅延 → races/entries/payouts を delete & re-insert → 末尾で「再投入後の races に
存在しない race_id」を odds_snapshots からだけ prune。

**結果**: realtime odds snapshot は rebuild で消えない。同じ race_id が再投入される
限り、odds_snapshots の対応行は残り続ける。

### 4. auto_paper_trading.py の cascade abort

`scripts/live/auto_paper_trading.py` は upstream step が fail したら downstream を
skip するように gating されている:

```
fetch fail → parse skip + generate skip + record skip + last_error 更新
parse fail → generate skip + record skip + last_error 更新
generate fail → record skip + last_error 更新
```

run_summary には `parse: "skip_upstream_failed"` と明示記録され、状態追跡が崩れない。

### 5. 既存 DB のバックアップ

毎回の rebuild 前に `data/backups/keiba_YYYYMMDD.sqlite` が作られる
(auto_paper_trading の日次 backup)。事故時は単純 copy で復元可能:

```bash
cp data/backups/keiba_YYYYMMDD.sqlite data/db/keiba.sqlite
```

---

## VPS 自動運用 (Linux + cron / systemd)

このプロジェクトは **VPS 常駐 cron 運用** を前提に設計されている。
ローカル PC や Claude Code を閉じても、VPS だけで毎週末自動で paper trading が回る。

### 1. なぜ VPS か

- **電源・ネット途切れの影響を受けない**: 自宅 PC では夜中の cron が回らない
- **Claude Code とは独立**: pipeline は subprocess 経由で既存 script を呼ぶだけ
- **journalctl / logs/ で履歴が残る**: 後追い・障害分析が容易
- **systemd timer で reboot 後も自動復帰**: `Persistent=true` で取りこぼし catch-up

### 2. Ubuntu 22.04 LTS 前提セットアップ

```bash
# === システム準備 ===
sudo apt update && sudo apt install -y python3.11 python3.11-venv git sqlite3
sudo timedatectl set-timezone Asia/Tokyo                    # ★ JST 強制
timedatectl                                                  # 確認

# === コード取得 ===
cd ~
git clone https://github.com/your-user/keiba-syutoku.git
cd keiba-syutoku

# === Python 仮想環境 ===
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# === DB 初期化 (既存 DB 持ち込みなら skip) ===
python scripts/init_db.py
# 過去データを fetch するなら:
#   python scripts/ingest/fetch_race_results.py --years 2024 2025 2026
#   python scripts/transform/parse_race_results.py --rebuild
# 既存 DB を VPS にアップロードするなら data/db/keiba.sqlite に置く

# === 動作テスト (dry-run) ===
python scripts/live/auto_paper_trading.py --dry-run
python scripts/live/auto_paper_trading.py --status

# === 手動 1 回実行 (本番に近い動作) ===
bash scripts/run_paper_trading_once.sh
tail logs/paper_trading_$(date +%Y%m%d).log
```

### 3. cron 方式 (シンプル)

```bash
bash scripts/install_cron.sh    # 登録
crontab -l                       # 確認
# */5 9-18 * * 6,0 /path/to/scripts/run_paper_trading_once.sh # KeibaPaperTrading

bash scripts/uninstall_cron.sh   # 解除
```

スケジュール = **毎週 土日, 09:00〜18:59, 5 分間隔**。
`cron` は system timezone を使うので、**事前に `sudo timedatectl set-timezone Asia/Tokyo` 必須**。

### 4. systemd timer 方式 (推奨)

```bash
bash scripts/install_systemd_timer.sh
systemctl --user list-timers
journalctl --user -u keiba-paper.service -n 50

# 手動 1 回実行
systemctl --user start keiba-paper.service

# 解除
systemctl --user disable --now keiba-paper.timer
rm ~/.config/systemd/user/keiba-paper.{service,timer}
systemctl --user daemon-reload
```

- **`Persistent=true`** で VPS 停止中に missed trigger を 1 回 catch-up
- ヘッドレス VPS では `sudo loginctl enable-linger $(whoami)` 推奨 (ログイン無しでも user unit が走る)
- service の `Environment=TZ=Asia/Tokyo` で確実に JST

### 5. 運用 / 確認コマンド

```bash
# 現在状態 + lock + log tail + scheduler 状態
python scripts/live/auto_paper_trading.py --status

# stale lock を強制解除 (4h 経過時は自動だが手動も可能)
python scripts/live/auto_paper_trading.py --force-unlock

# DB backup 強制実行 (日次の自動 backup を待たずに走らせたい時)
python scripts/live/auto_paper_trading.py --backup-db

# logs 確認
tail -f logs/paper_trading_$(date +%Y%m%d).log
tail -f logs/errors_$(date +%Y%m%d).log

# backup 確認
ls -lh data/backups/

# state 確認
cat data/processed/pipeline_state.json
```

### 6. lock / state / backup の仕様

| 仕組み | 場所 | 内容 |
|---|---|---|
| **lock** | `data/processed/.paper_trading.lock` | JSON `{pid, acquired_at, host}`。4h 経過で stale 自動解除。`--force-unlock` で手動解除 |
| **state** | `data/processed/pipeline_state.json` | JSON `{last_fetch, last_parse, last_generate, last_record, last_backup, last_success, last_error, runs_total, runs_successful, runs_failed, history}`。毎 run で更新 |
| **backup** | `data/backups/keiba_YYYYMMDD.sqlite` | DB を日次 copy。同日 backup 既存なら skip |
| **logs** | `logs/paper_trading_YYYYMMDD.log` + `logs/errors_YYYYMMDD.log` | 日次 split, stdout/stderr 分離 |

### 7. VPS 再起動後の挙動

| シナリオ | cron | systemd |
|---|---|---|
| reboot 直後 | 次の 5 分境界で再開 | **`Persistent=true` で missed run を 1 回 catch-up** |
| network 切断中の trigger | スキップ (cron daemon 上は 1 回失敗) | スキップ + missed mark |
| disk full | log 書き失敗、lock 残る | journalctl に記録、lock 残る |
| Python crash | exit ≠ 0 → log/err に記録、lock は finally で解放 | 同じ + journal にも残る |

### 8. timezone 厳守

- Python 側: `ZoneInfo("Asia/Tokyo")` で全 datetime tz-aware
- shell 側: `export TZ="Asia/Tokyo"` を `run_paper_trading_once.sh` で setting
- cron / systemd 側: system tz が JST であること (`timedatectl` で確認)
- 3 つすべて JST に揃えないと「土日朝の cron が金曜深夜に動く」などの事故が起きる

### 9. Claude Code を閉じても動く理由

```
Claude Code  ←─ (人) ─→ git push
                          ↓
                       VPS git pull
                          ↓
            cron / systemd timer (= OS daemon)
                          ↓
            run_paper_trading_once.sh  ← 5 分おき自動 kick
                          ↓
            auto_paper_trading.py  ← single-run, lock 取得 → 必要処理 → 終了
                          ↓
            既存 strategy E_D9_P3_CAP4 ロジック (subprocess)
                          ↓
            DB / logs / backup
```

- Claude Code は **コード書き換えの道具に過ぎない**。一度 git push したら関係ない
- ローカル PC は **コード編集にしか使わない**。閉じても VPS の cron は OS daemon として動く
- VPS が動いていれば pipeline は永久に回る

### 10. よくあるトラブル

| 症状 | 原因 | 対処 |
|---|---|---|
| 「lock held by pid=X」エラーで毎回 skip | 前回 run が異常終了して lock 残った | 4h 経てば自動解除。即なら `python scripts/live/auto_paper_trading.py --force-unlock` |
| cron が動かない | `crontab -l` に entry ない、または system TZ が違う | `bash scripts/install_cron.sh` 再実行 + `timedatectl` 確認 |
| logs/ が増殖し続ける | rotation してない | `find logs/ -name "*.log" -mtime +30 -delete` を別 cron に |
| backup ディスクが埋まる | 日次 backup 蓄積 | `find data/backups/ -name "*.sqlite" -mtime +90 -delete` 検討 |
| fetch が失敗続き | netkeiba 側 outage / IP block | `state.last_error` 確認、polite_sleep 確認、後日再 run |
| Python 仮想環境が壊れる | apt upgrade 等で Python メジャーバージョン変更 | `rm -rf .venv && python3.X -m venv .venv && pip install -r requirements.txt` |

### 11. ⚠️ 注意

- **既存 strategy E_D9_P3_CAP4 のロジックは触っていない**。auto_paper_trading は subprocess で既存 script を呼ぶだけ
- 実投票はしない。「もし買っていたら」を log に積むだけ
- 半年〜1 年 paper 運用 → 実際の ROI と backtest 期待値 (+278% / 年 +25K) を比較してから実投票を検討

### 12. VPS 初回デプロイ手順 (完全コピペ可能)

Ubuntu 22.04 LTS を新規 VPS に立てた直後から、systemd timer が回り始めるまで。
**所要時間: 約 15 分** (fetch を含めない場合)。

```bash
# === [1/15] OS update + 必要パッケージ ===
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git sqlite3 tzdata

# === [2/15] timezone を JST に強制 ===
sudo timedatectl set-timezone Asia/Tokyo
timedatectl                                            # → Time zone: Asia/Tokyo (JST, +0900) を確認

# === [3/15] git clone ===
cd ~
git clone https://github.com/your-user/keiba-syutoku.git
cd keiba-syutoku

# === [4/15] venv 作成 ===
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# === [5/15] requirements install ===
pip install -r requirements.txt

# === [6/15] .env 配置 (必要があれば) ===
# cp .env.example .env       # 現状 .env は必須ではないが、将来追加された場合
# vi .env                     # API key 等を埋める

# === [7/15] DB 配置 ===
# パターン A: 手元 DB を scp で持ち込む (過去 1800+ race の蓄積を活かす)
#   ローカル PC で:  scp data/db/keiba.sqlite vps:~/keiba-syutoku/data/db/
# パターン B: VPS で fetch から作り直す (時間がかかる)
#   python scripts/init_db.py
#   python scripts/ingest/fetch_race_results.py --years 2024 2025 2026
#   python scripts/transform/parse_race_results.py --rebuild

# === [8/15] timezone × Python 整合性確認 ===
python -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('Asia/Tokyo')))"

# === [9/15] first-run setup wrapper を実行 (chmod / mkdir / dry-run / status / deploy_check) ===
bash scripts/first_run_vps.sh

# === [10/15] 単発 dry-run (念のため再確認) ===
python scripts/live/auto_paper_trading.py --dry-run
python scripts/live/auto_paper_trading.py --status

# === [11/15] systemd timer install (推奨) ===
bash scripts/install_systemd_timer.sh

# === [12/15] timer がスケジュール上に乗ったか確認 ===
systemctl --user list-timers | grep keiba-paper
systemctl --user status keiba-paper.timer --no-pager

# === [13/15] ヘッドレス VPS で持続させる (login session 不要にする) ===
sudo loginctl enable-linger "$(whoami)"

# === [14/15] 試しに 1 回 kick (本物の pipeline) ===
systemctl --user start keiba-paper.service
sleep 5
journalctl --user -u keiba-paper.service -n 30 --no-pager
tail -n 30 logs/paper_trading_$(date +%Y%m%d).log

# === [15/15] backup が作成されたか確認 ===
ls -lh data/backups/
```

✅ ここまで全部 PASS なら本番運用準備完了。
週末を待たずに動作確認したいなら手動 `systemctl --user start keiba-paper.service` を何度か実行。

### 13. 本番前チェック (運用開始の最終ゲート)

**この 7 項目を毎回確認してから timer を on にする** (= 移行後・大きな変更後の sanity check)。

```bash
# 1. DB の各テーブル row count
sqlite3 data/db/keiba.sqlite '
  SELECT "races" AS t, COUNT(*) FROM races
  UNION ALL SELECT "entries", COUNT(*) FROM entries
  UNION ALL SELECT "payouts", COUNT(*) FROM payouts
  UNION ALL SELECT "odds_snapshots", COUNT(*) FROM odds_snapshots;'

# 2. DB validate (schema / FK / NULL / 異常値)
python scripts/analysis/validate_db.py

# 3. 全 pipeline の dry-run (DB 不変)
python scripts/live/auto_paper_trading.py --dry-run

# 4. バックアップを 1 個作成 (移行前 snapshot)
python scripts/live/auto_paper_trading.py --backup-db
ls -lh data/backups/ | tail -3

# 5. lock が無いことを確認
ls -la data/processed/.paper_trading.lock 2>/dev/null && \
    echo "WARN: lock present" || echo "OK: no lock"

# 6. timer の次回起動時刻
systemctl --user list-timers --no-pager | grep keiba-paper

# 7. 直近 journal の error が無いか
journalctl --user -u keiba-paper.service --since "1 hour ago" --no-pager | tail -20
```

すべて緑なら **`systemctl --user enable --now keiba-paper.timer`**。

### 14. 運用中の確認コマンド (週次の眺める set)

```bash
# 現在の状態 1 行サマリ (lock / state / 直近 run / scheduler)
python scripts/live/auto_paper_trading.py --status

# 今日の log を follow (土日の本番中はこれを見る)
tail -f logs/paper_trading_$(date +%Y%m%d).log
tail -f logs/errors_$(date +%Y%m%d).log

# 今日の journal を follow
journalctl --user -u keiba-paper.service -f

# timer 全体の健康状態
systemctl --user status keiba-paper.timer --no-pager
systemctl --user list-timers --no-pager | head

# backup を時系列で並べる (日次で増えているはず)
ls -lh data/backups/ | tail -10

# state JSON を pretty print
python -m json.tool data/processed/pipeline_state.json

# paper trading の累積成績
tail -20 data/processed/paper_trading_log.csv
```

### 15. 緊急停止・復旧 (障害時 playbook)

#### 15-A. 即時停止 (例: 大きな bug を見つけた・netkeiba から block された)

```bash
# timer を即 stop + 起動も無効化
systemctl --user disable --now keiba-paper.timer

# 進行中の run があれば kill
pkill -f auto_paper_trading.py 2>/dev/null || true

# lock を強制解除 (次回 enable 時に hung lock で詰まらないよう)
python scripts/live/auto_paper_trading.py --force-unlock
```

#### 15-B. DB を直近 backup から復旧

```bash
# 1. 念のため現状 DB を退避
mv data/db/keiba.sqlite data/db/keiba.sqlite.broken_$(date +%Y%m%d_%H%M%S)

# 2. 直近 backup から復旧
LATEST_BACKUP="$(ls -t data/backups/keiba_*.sqlite | head -1)"
echo "restoring from: ${LATEST_BACKUP}"
cp "${LATEST_BACKUP}" data/db/keiba.sqlite

# 3. integrity / schema を validate
sqlite3 data/db/keiba.sqlite 'PRAGMA integrity_check;'   # 'ok' なら OK
python scripts/analysis/validate_db.py                    # verdict OK を確認

# 4. dry-run で動作確認
python scripts/live/auto_paper_trading.py --dry-run
```

#### 15-C. 復旧 → timer 再開

```bash
# 1. deploy_check で全項目 PASS を確認
bash scripts/deploy_check.sh

# 2. 1 回手動 kick して journal で確認
systemctl --user start keiba-paper.service
journalctl --user -u keiba-paper.service -n 50 --no-pager

# 3. timer を再 enable
systemctl --user enable --now keiba-paper.timer
systemctl --user list-timers | grep keiba-paper
```

#### 15-D. 完全アンインストール (VPS 整理時)

```bash
bash scripts/install_systemd_timer.sh --uninstall
bash scripts/uninstall_cron.sh                      # cron 方式併用していた場合
# (オプション) repo / venv も削除
# cd ~ && rm -rf keiba-syutoku
```

### 16. Telegram 通知の設定

VPS 上で動く pipeline の **start / ticket 生成 / 結果記録 / error** を手元の
Telegram に通知する。**Bot Token / Chat ID は `.env` に保存** し、リポジトリには
絶対に commit しない (`.env` は `.gitignore` で除外済み)。

#### 16-1. BotFather で Bot を作る (手元 PC の Telegram アプリで)

1. Telegram で **@BotFather** を検索して開く
2. `/start` → `/newbot`
3. **Bot の表示名** を聞かれる → 例: `keiba-syutoku-notify`
4. **Bot の username** を聞かれる (末尾 `_bot` 必須) → 例: `keiba_syutoku_notify_bot`
5. 成功すると BotFather が **token** を返す:
   ```
   123456789:ABCdefGHIjklMNOpqrstuvWXYZ-0123456789
   ```
   ↑ これが `TELEGRAM_BOT_TOKEN`。**他人に絶対見せない**。

#### 16-2. Bot に話しかけて Chat ID を取得

token を取った直後の bot は、まだあなたとの会話履歴がゼロ。**Chat ID** を取得するには:

1. Telegram で自分の bot を開く (BotFather のメッセージに `t.me/<username>_bot` リンクがある)
2. **`/start` を bot に送る** (本文は何でも良い、`hi` でも可)
3. 手元 PC のブラウザで以下を開く (TOKEN を貼り替え):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. 返ってきた JSON の `"chat":{"id":123456789,...}` の **数字** が `TELEGRAM_CHAT_ID`
   ```json
   {"ok":true,"result":[{"message":{"chat":{"id":987654321,"first_name":...}}}]}
   ```

メモ:
- `getUpdates` は **直近 24 h** のメッセージしか返さない。空 (`"result":[]`) なら bot に `/start` をもう一度送る
- グループに通知したい時は bot をグループに招待し、グループ内で何か発言 → `id` は **負の値** (例 `-1001234567890`) になる

#### 16-3. VPS で `.env` を作る

VPS に SSH している状態で:

```bash
cd ~/keiba-syutoku
cp .env.example .env
nano .env
```

中身をこう書き換える (3 行だけ):
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrstuvWXYZ-0123456789
TELEGRAM_CHAT_ID=987654321
ENABLE_TELEGRAM=true
```

権限を絞っておく (他ユーザーから token が読まれないように):
```bash
chmod 600 .env
ls -l .env       # -rw------- (600) になっていれば OK
```

> ⚠️ `git status` で `.env` が "Untracked" 扱いされてないことを確認。表示される場合は
> `.gitignore` を確認 (デフォルトで除外されている)。

#### 16-4. 動作確認

```bash
source .venv/bin/activate    # まだ activate してなければ
python scripts/live/test_telegram.py --message "hello from VPS"
```

期待される出力:
```
=== Telegram test ===
  ENABLE_TELEGRAM    : true
  TELEGRAM_BOT_TOKEN : set (len=46, hidden)
  TELEGRAM_CHAT_ID   : 987654321

sending: 'hello from VPS'
[OK] delivered
```

Telegram アプリに **"hello from VPS"** が届けば成功。

exit code:
- `0` 配送成功
- `1` 配送失敗 (token / chat_id は埋まってるが API がエラー)
- `2` `.env` が埋まっていない / `ENABLE_TELEGRAM=false`

#### 16-5. pipeline 通知が届くか dry-run

```bash
python scripts/live/auto_paper_trading.py --dry-run
```

dry-run は **通知を送らない** (誤通知防止)。実通知は本番 run で:
```bash
python scripts/live/auto_paper_trading.py
# または
systemctl --user start keiba-paper.service
```

実 run で届く通知 (1 日 1 回ずつ dedupe):
| タイミング | 内容例 |
|---|---|
| pipeline 開始 | `🏇 keiba paper trading: pipeline started\n  date: ...` |
| 買い目生成 | `🎫 tickets generated\n  file: live_tickets_YYYY-MM-DD_final.csv (N tickets)` |
| 結果記録 | `📊 results recorded\n  log: paper_trading_log.csv` |
| エラー時 | `⚠️ keiba pipeline FAILED: <step>\n  detail: ...\n  see logs/errors_*.log` |

#### 16-6. 重複通知の仕組み (なぜ 5 分 cron で spam しないか)

`data/processed/telegram_sent.json` に `(race_id, snapshot_time, strategy, notification_type)`
の組を JSON で保存。同じ key の通知は **2 度送られない**。
- pipeline 系: `race_id = YYYY-MM-DD`, `snapshot_time = "pipeline"`, `notification_type = "start"|"generate"|"record"`
- error 系: `notification_type = "error_<sha1_8>"`  → **異なる error は別通知**、**同じ error は 1 回**

クリア (=全通知を再送可能にする):
```bash
rm data/processed/telegram_sent.json
```

retention は 30 日 — それより古い entry は次回保存時に自動 prune。

#### 16-7. 通知が来ない時のチェックリスト

| 症状 | 確認 / 対処 |
|---|---|
| `test_telegram.py` が `[FAIL] ENABLE_TELEGRAM is not 'true'` | `.env` の `ENABLE_TELEGRAM=true` (大小区別なし)、値の前後に空白なし |
| `[FAIL] TELEGRAM_BOT_TOKEN is not set` | `.env` が repo root にある? `cd ~/keiba-syutoku && ls -la .env` で確認 |
| HTTP 401 Unauthorized | token が古い or 無効。BotFather で `/mybots → 該当 bot → API Token` で再取得 |
| HTTP 400 chat not found | chat_id が間違い。bot に `/start` を再送 → `getUpdates` でやり直し |
| HTTP 403 Forbidden | bot をブロックしてる。Telegram で bot 画面 → unblock |
| Bot に `/start` を送っても getUpdates が空 | bot username の typo / 別の bot 開いてる |
| `requests not installed` warning | venv が activate されてない or `pip install -r requirements.txt` 未実行 |
| pipeline は走ってるのに通知ゼロ | dedupe が効いてる可能性 → `rm data/processed/telegram_sent.json` で reset |
| token を一度 commit してしまった | **即 BotFather で `/revoke`** → 新 token を `.env` に書き直し → git history は `git filter-repo` 等で消す |

#### 16-8. 通知を一時的に止める (token は残したまま)

```bash
sed -i 's/^ENABLE_TELEGRAM=.*/ENABLE_TELEGRAM=false/' .env
# 次の cron tick から通知が止まる (process は走り続ける)
```

再開:
```bash
sed -i 's/^ENABLE_TELEGRAM=.*/ENABLE_TELEGRAM=true/' .env
```

| 症状 | first step |
|---|---|
| timer が動いていない | `systemctl --user list-timers \| grep keiba` → 居なければ install_systemd_timer.sh 再実行 |
| lock が外れない | `python scripts/live/auto_paper_trading.py --force-unlock` |
| DB が壊れた疑い | `sqlite3 data/db/keiba.sqlite 'PRAGMA integrity_check;'` → ok 以外なら 15-B |
| parse が毎回 fail | `tail logs/errors_*.log` → FK エラーなら #49 (parse_race_results.py の defer_foreign_keys を確認) |
| fetch が空 | `logs/errors_*.log` で 403/429 確認 → 数時間待つ。`scripts/ingest/fetch_race_results.py --years YYYY` を手動で 1 回 |
| backup が増えすぎ | `find data/backups/ -name "*.sqlite" -mtime +90 -delete` |
| logs が増えすぎ | `find logs/ -name "*.log" -mtime +30 -delete` を別 cron で |

---

E ルールはリアルタイム運用するなら「発走 30 分前 odds」など pre-race snapshot で判断
する必要がある (確定 odds とは popularity 順位が入れ替わる)。そのための時系列 odds 表。

### スキーマ

```
odds_snapshots
  id                   PK (autoincrement)
  race_id              FK → races.race_id
  snapshot_time_label  "60min" / "30min" / "10min" / "5min" / "final" 等
  captured_at          実際に odds を観測した wall-clock time
  horse_number
  popularity
  win_odds
  source               "netkeiba_realtime" / "db_final_odds" /
                       "db_final_odds_PLACEHOLDER" (= 暫定 fallback)
  created_at           INSERT 時刻
  UNIQUE (race_id, snapshot_time_label, horse_number)
```

Index: `(race_id, snapshot_time_label)` と `(race_id, horse_number)` の 2 本。

### 取得フロー

```powershell
# label "final" を entries の確定 odds から転記 (本物の odds source)
python scripts\ingest\fetch_odds_snapshots.py --year 2025 --snapshot-time final

# label "30min" 等を投入 (現状は realtime fetcher 未実装のため
# entries の確定 odds を PLACEHOLDER として使い、source 列で識別)
python scripts\ingest\fetch_odds_snapshots.py --year 2025 --snapshot-time 30min

# 既に同 (race_id, snapshot_time_label) が存在するなら skip。--force で置換。
```

### generate_tickets.py の fallback 仕様

`--snapshot-time` 指定時、odds の取得優先順位:

1. **`odds_snapshots` テーブルにその snapshot_time_label の row があれば** その popularity/win_odds を使用 → `odds_source = "db_snapshot"`
2. **無ければ** entries テーブルの確定 odds に fallback → `odds_source = "db_final_fallback"`

両者は出力 CSV の `odds_source` 列で per-race に区別される。
**`db_final_fallback` のレースは「事前 odds が無く確定 odds で代用した」ことを意味するため、リアルタイム運用では信頼してはならない。**

### Coverage 確認

```powershell
python scripts\analysis\validate_snapshot_coverage.py
# → data\processed\odds_snapshot_coverage.csv
```

snapshot_time_label 別に races / horses / coverage_pct / NULL率 / 重複 / 不足 を出す。
新しい label を追加したら必ず実行して欠損が無いことを確認。

### リアルタイム化時の注意点

1. **netkeiba realtime fetcher は未実装**: `scripts/ingest/fetch_odds_snapshots.py` の
   `fetch_odds_for_race()` を realtime source 実装に差し替える必要あり (現状は entries
   から転記するだけの PLACEHOLDER)。
2. **発走前 odds は変動する**: 30 分前 → 10 分前 → final で popularity 順位が入れ替わる
   ことは普通。`compare_snapshot_vs_final.py` で「snapshot 採用時の hit と final 採用時の
   hit がどれだけズレるか」を継続観測すべき。
3. **欠場・除外馬の扱い**: 発走前 snapshot を取った後にスクラッチが出ると、後で取った
   snapshot とで horse 数が変わる。validator の `races_with_missing_horses` で検知。
4. **取得間隔**: realtime fetcher は短間隔で叩くと JRA/netkeiba 側に負荷をかける。
   1 race あたり 1 snapshot に絞り、`FETCH_SLEEP_SECONDS` を守ること。
5. **odds_snapshots を growing table として扱う**: パース dump (`parse_race_results.py
   --rebuild`) は races/entries/payouts のみを破壊する。`odds_snapshots` は別管理。
   再構築したい場合は `python scripts\init_db.py --reset` で全テーブルを wipe してから
   parse + fetch を回す。

### ⚠️ 注意点

1. **過去データ上の検証結果であり将来の利益を保証するものではない**: backtest は 2016-2025
   の 10 年データに基づく。馬産業・JRA 制度・出走馬の傾向は変動するため、未来も同じ ROI に
   なるとは限らない。
2. **オッズは「最終 (確定) オッズ」ベース**: 本 DB に入っている win_odds は発走時点の確定値。
   リアルタイム購入では締切前の暫定オッズで判断するため、dark horse 選定がズレる可能性あり。
   特に「win_odds 10-30」境界付近の馬は当日変動で対象が入れ替わる。
3. **まずは実投票せず paper trading 推奨**: 半年〜1 年実際の generate→record サイクルを
   無投票で回し、backtest 期待値と乖離していないか確認してから実運用へ。
4. **bet サイズは絶対に固定**: 「直近不調だから増やす」「絶好調だから倍プッシュ」は
   bankroll simulation で破産確率が跳ね上がることが確認済み。1 点 100円固定を守る。
5. **負け年がある**: 10年中 4-5 年が年単位赤字。**単年で諦めず、5 年スパンで評価**。
6. **2023 年は全 component 全敗**: 統計上 1/10 確率で全敗年がある。drawdown 30% を覚悟する。

---

## アーキテクチャの考え方

| 層 | 役割 | 差し替えのしかた |
|---|---|---|
| **ingest** | 公開サイトから生HTMLを取ってきてキャッシュするだけ | `scripts/ingest/base.py` の `BaseFetcher` を実装した新クラスを足し、`fetch_race_results.py:get_fetcher()` に登録 |
| **transform** | 生HTML → 構造化（races/entries/payouts） | `scripts/transform/base.py` の `BaseParser` を実装した新クラスを足し、`parse_race_results.py:get_parser()` に登録 |
| **DB** | 単一のSQLite。race_idでJOIN可能 | `src/schemas.py` を編集 → `init_db.py` 再実行 |
| **backtest** | 事前情報のみで買い目生成 → 三連単払戻と完全一致で的中判定 | `scripts/backtest/simple_roi.py:STRATEGIES` に関数を追加 |

### 設計上の方針（あとから効いてくる）

- **raw / processed の分離**: スクレイピング結果は `data/raw/` に置きっぱなしにし、整形後のデータは `data/processed/` と DB に置く。パース側のバグはraw HTMLを使って何度でも再現できる。
- **取得は冪等**: HTMLが既にあるならネットワークを叩かない。crawl中断 → 再開がいつでもできる。
- **未来情報リーク防止**: バックテストの買い目生成は `popularity` / `win_odds` / `horse_number` だけを使う。`finish_position` は的中判定にしか使わない。
- **ソースは差し替え前提**: ingest/transform とも `BaseFetcher` / `BaseParser` に対するプログラミング。netkeibaから JRA-VAN / 自前データに差し替えるときも `src/schemas.py` 以下は触らない。
- **アクセス間隔を縮めない**: `FETCH_SLEEP_SECONDS` 環境変数で変更できるが、3秒未満には設定しないこと。

---

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `FETCH_SLEEP_SECONDS` | `3.0` | リクエスト間スリープ秒 |
| `FETCH_TIMEOUT_SECONDS` | `20.0` | HTTPタイムアウト |
| `FETCH_USER_AGENT` | identifying string | User-Agent ヘッダ |
| `KEIBA_SOURCE` | `netkeiba` | 取得元の識別子 |

---

## 今後の拡張方針

### 短期
- [ ] パーサ精度向上（グレード判定・障害レース対応・複数の払戻表記揺れ）
- [ ] race_id discovery の安定化（netkeibaのHTML構造変更に追従するテスト）
- [ ] バックテストに **券種別ROI**（馬連・ワイドなど）の比較を追加
- [ ] 払戻分布の可視化 notebook を `notebooks/` に追加

### 中期
- [ ] 過去15〜30年分への対象拡張（取得スクリプトの差分実行を確認）
- [ ] 馬・騎手・調教師など master テーブル追加
- [ ] 出走表（前日オッズ含む）も取得する `fetch_race_cards.py` を追加 → 真の事前情報のみでバックテスト可能に
- [ ] テストデータ（小さな固定HTML）でのパーサユニットテスト

### 長期
- [ ] AI予測モデルの追加（このDB上にFeature Storeを作る形）
- [ ] 期待値 > 1.0 の買い目だけを抽出する「閾値バックテスト」
- [ ] パドック・調教情報など別ソースの統合
