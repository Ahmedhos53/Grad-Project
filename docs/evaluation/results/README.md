# Evaluation results

I keep these result files alongside the report so I can trace its figures and measurements back to their recorded inputs. The files contain public-trip predictions, synthetic rule checks, and aggregate resource measurements. They contain no raw camera or microphone media, transcript text, or private trip recordings.

## Telemetry and fusion

I use `continuous_telemetry_crossvalidated.json` for the corrected trip-held-out predictions and `threshold_sensitivity_crossvalidated.json` for the descriptive confidence sweep. The frozen encoder stays unchanged while each fold fits its head on the other trips. I use `fusion_sensitivity_evaluation.json` to examine synthetic score mechanics and signal dependencies; those results are not detection accuracy.

## Resource measurements

I retain the three `combined_resource_stability_900*.json` runs and their summary. These describe bounded runs on one laptop, including the selected stage timings and queue counters. The paths inside these recorded files identify their original local inputs and outputs; they do not imply that generated files are already present in a fresh checkout.

## Rebuilding the figures

I can rebuild the plots from these retained results without loading a model or opening a camera or microphone:

```powershell
python tools/build_evaluation_figures.py --evaluation-dir docs/evaluation/results --telemetry-report docs/evaluation/results/continuous_telemetry_crossvalidated.json --threshold-report docs/evaluation/results/threshold_sensitivity_crossvalidated.json --fusion-report docs/evaluation/results/fusion_sensitivity_evaluation.json --output-dir outputs/rebuilt_figures
```

I keep the rebuilt plots under `outputs/` so a local run does not overwrite the report figures. Library versions can change rendering details, so I check the plots against the retained measurements.

## Synthetic visual and audio checks

I retain `visual_logic_evaluation.json` and `audio_protocol_evaluation.json` as software checks. They use constructed examples and do not establish real visual precision/recall or speech-recognition WER/CER. The labelled visual and consented-audio protocols are separate.
