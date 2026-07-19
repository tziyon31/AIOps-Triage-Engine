# Promotion Status Report

- Generated at: `2026-07-15T11:44:38.555081+00:00`
- Experiment: `log-triage-decision-engine`

## Status Counts

| candidate_status | count |
|---|---:|
| `missing` | 6 |
| `not_evaluated` | 30 |
| `selected` | 1 |
| `superseded` | 4 |

## Status Policy Checks

- Overall status: `passed`
- Single current candidate: `True`
- Selected runs have passed evidence: `True`
- Rejected runs have reason: `True`
- Current candidate run IDs: `['8e36e5abc3df49cfabcc0f1bb3fb5c2c']`

## MLflow Filter Examples

- `selected`: `tags.candidate_status = 'selected'`
- `eligible`: `tags.candidate_status = 'eligible'`
- `rejected`: `tags.candidate_status = 'rejected'`
- `superseded`: `tags.candidate_status = 'superseded'`
- `not_evaluated`: `tags.candidate_status = 'not_evaluated'`
- `current_candidate`: `tags.current_candidate = 'true'`
- `candidate_history`: `tags.candidate = 'true'`

## Runs

| Status | Variant | Run ID | Candidate | Current | Evidence | Reason |
|---|---|---|---|---|---|---|
| not_evaluated | manual_tfidf_sgd_log_loss | 1db2300c064843a394f67486cf226e7c | false | false | passed | not_evaluated |
| selected | manual_tfidf_logistic_regression | 8e36e5abc3df49cfabcc0f1bb3fb5c2c | true | true | passed | candidate_policy_and_baseline_passed |
| not_evaluated | manual_tfidf_sgd_log_loss | a848c91f95a14727b521c1c319ed581f | false | false | passed | not_evaluated |
| superseded | manual_tfidf_logistic_regression | 7bfd8e17403a4984a625fa4b007063e0 | true | false | passed | candidate_policy_and_baseline_passed |
| missing | unknown | d7e028c13e5c404eb255df29a9be737a | false | false | failed | missing |
| not_evaluated | manual_tfidf_logistic_regression | c359e014e84040fe8f9335b8cb37544c | false | false | passed | not_evaluated |
| not_evaluated | manual_tfidf_sgd_log_loss | 03ec63bbe77e4e649326d79b7881a6c1 | false | false | failed | not_evaluated |
| not_evaluated | manual_tfidf_logistic_regression | 710206f57ec84252b73e952ae239a45e | false | false | failed | not_evaluated |
| superseded | tuned_combined_candidate | ca99ebe55172433a9bfa2d20b4a6bd73 | true | false | failed | candidate_policy_and_baseline_passed |
| not_evaluated | current_baseline_config | 20e0daf5425440ec832b9b089f86ab6f | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_sgd_log_loss | c0933cf9f5fe48fb9d91e91f999349a4 | false | false | failed | promotion_gate_not_implemented_yet |
| superseded | manual_tfidf_logistic_regression | 40fa1b4f96544a76a86ac13d82986c33 | true | false | failed | candidate_policy_and_baseline_passed |
| not_evaluated | manual_tfidf_sgd_log_loss | b771ef74c0f44d22bc5317d0b862bc06 | false | false | failed | promotion_gate_not_implemented_yet |
| superseded | manual_tfidf_logistic_regression | fbf5b49b832a48f1ad057e1601453630 | true | false | failed | candidate_policy_and_baseline_passed |
| not_evaluated | manual_tfidf_sgd_log_loss | 9913c2aaeb7e41e5b25e832cd650340d | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_logistic_regression | eb248f38e0c645e7b2aa07181dcd3a89 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_sgd_log_loss | 4e7e9142a00c4362b22a8a03fad5a924 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_logistic_regression | 6eb3fb2da2fc45ea8470b861b9408562 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_sgd_log_loss | bde8c1331b6a4f6bab0a5bdda39b3a6a | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_logistic_regression | 0bf8e629ec634882b245035bf84a57d6 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_sgd_log_loss | a3045cc2073b46818de7f0f50bd569d9 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_logistic_regression | 246e805b2fb142f79618bc01af4540e9 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | manual_tfidf_logistic_regression | 78658abeeede47969c56c64fabd49974 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | e06ad25bd3e943e5b63301d3bea02f4c | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | bbfd273bf3574f3787ec8578fee55524 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 7286d9707eee4230ace8842531b65883 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 47e6f9fd939d4a6a8bc4ceb7355873ed | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | dc6312230b354ac69fec2d26d0d9c503 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 3cbb982b0bb44e93a8333f6068fcbda6 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | e962665619b8495bbeebfa8d8954f83b | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 033d4e7c80ed400ebafa589ae8f30da7 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 2bb8102ae4f94b97aea3832c8a041116 | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 4fb7cfc49b734f20acce81530c7b18ac | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 36e1eab2c8f64889b1316137f9852feb | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | f15cc433e9f94379ac6cbd2bd09408da | false | false | failed | promotion_gate_not_implemented_yet |
| not_evaluated | unknown | 0d6ff27c1ffc48b2b1bbbd763db8bafd | false | false | failed | promotion_gate_not_implemented_yet |
| missing | unknown | fc4575c73a014cd1a36dcecd0b0c4f71 | false | false | failed | missing |
| missing | unknown | 3d190b902b714052aeb57bcae355e8b3 | false | false | failed | missing |
| missing | unknown | 85dafd28b0ca4659ae5c39aa7ae44253 | false | false | failed | missing |
| missing | unknown | 8da731e3842a4d5d829f5a4d0c685156 | false | false | failed | missing |
| missing | unknown | 86cf9edae6bc43058ffc18601de937ec | false | false | failed | missing |

## Boundary

This report summarizes promotion status visibility in MLflow.
It does not select, reject, or promote runs.
