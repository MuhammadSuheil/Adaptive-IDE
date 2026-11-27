# Dataset yang Dibaca Notebook Pemodelan

| File | Isi | Dibaca oleh |
|---|---|---|
| `v1_active_task/good_acceptable.csv` | V1, window 25 s, label easy/medium/hard | Batch 01 (old-dataset) |
| `v2_session_phase/window_{15,25,45}s_step_5s/good_only.csv` | V2 view good_only | Batch 01 (new-dataset) |
| `v2_session_phase/window_{15,25,45}s_step_5s/good_acceptable.csv` | V2 view good_acceptable (target Low/Medium/High) | Batch 01 (new-dataset), Batch 02–05; Batch 06 `v3/exp01–02` (25 s); Batch 06 `v3/exp06–07` (15 s, riwayat sensor) |
| `v2_session_phase/window_{15,25,45}s_step_5s/all_data.csv` | V2 seluruh window berlabel | Batch 01 (new-dataset); 15 s: Batch 06 `v3/exp06–13`, Batch 07, Batch 08 (kandidat window, riwayat sensor, prior auxiliary); 45 s: Batch 06 `v3/exp12` |
| `v3_behavioral_proxy/episode_labels.csv` | Bukti episode dan label V3 | Batch 06, Batch 07, Batch 08 exp04 |
| `v3_behavioral_proxy/participant_splits.csv` | Split peserta V3 (Train/validation/Test) | Batch 06, Batch 07, Batch 08 exp04 |
| `v3_behavioral_proxy/window_15s_step_5s/good_acceptable.csv` | Window V3 15 s | Batch 06, Batch 07, Batch 08 exp04 |
| `v3_behavioral_proxy/window_15s_step_5s/window_assignments.csv` | Pemetaan window → episode dan split | Batch 06, Batch 07, Batch 08 exp04 |
| `v3_behavioral_proxy/window_15s_step_5s/train_sample_weights.csv` | Bobot sampel Train V3 | Batch 06, Batch 07 |
| `v4_self_reported_rating/episode_ratings.csv` | Episode dan rating peserta | Batch 06, Batch 07, Batch 08 |
| `v4_self_reported_rating/window_15s_step_5s/good_acceptable.csv` | Window V4 15 s | Batch 06, Batch 07, Batch 08 |
| `v4_self_reported_rating/window_15s_step_5s/window_assignments.csv` | Pemetaan window → episode | Batch 06, Batch 07, Batch 08 |

Catatan:


