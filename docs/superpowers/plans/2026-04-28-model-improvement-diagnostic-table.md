# Model Improvement Diagnostic Table Implementation Plan

## Steps

1. Add unit tests for `build_model_improvement_diagnostics(...)`.
   - Create fixture timelines with one captured event, one noisy alert, and one missed event.
   - Assert group rows, count fields, risk summaries, z-feature rates, top feature counts, and recommendations.
   - Add a regression where the missed event has post-event rows that must not enter the pre-event summary.

2. Add alert-runner integration assertions.
   - Extend the alert episode experiment test to expect `model_improvement_diagnostics.csv`, `model_improvement_diagnostics.json`, and `model_improvement_diagnostic_report.md`.
   - Assert the JSON metadata and report title.

3. Implement `model_diagnostics.py`.
   - Build self-contained event extraction, captured-event matching, group summarization, and JSON-safe value helpers.
   - Keep the module deterministic and independent from report writing.

4. Wire diagnostics into `run_alert_episode_experiment(...)`.
   - Build the diagnostics after episode quality and case review.
   - Write CSV, JSON, and Markdown report artifacts.

5. Verify.
   - Run targeted tests.
   - Run `python -m pytest`.
   - Run the live command:
     `risk-engine --from-live-sources --paths-config config/paths.local.yaml --output-dir outputs --experiment-id model_improvement_diagnostics_v1 --alert-episodes --model-variant l2 --graph-window-size 4`

6. Update project docs.
   - Add the completed sprint and live interpretation to `AGENTS.md`.
   - Update `README.md` artifact/status sections.

7. Commit and push.
