# CabInspector

I built CabInspector as my graduation project: a local prototype for reviewing driver behaviour and ride quality. It combines a live cabin camera, optional microphone analysis, and replay of public trip telemetry in one dashboard.

I use pretrained models to produce observations. My rules then combine object positions, landmarks, persistence, and confidence into source-specific alerts and a bounded review score. A person remains responsible for interpreting those alerts.

## Features

I combine these features in the dashboard.

- Visual cues for eye state, movement outside a calibrated safe zone, phone context, and drinking context.
- Sound-event context, calibrated voice levels, and Arabic/English transcription.
- Public telemetry replay using a frozen PRIMUS encoder and a fitted classifier, with Random Forest available as a comparison.
- A dashboard showing source states and risk contributions, plus a timestamped CSV event log.
- Separate transcript display and storage controls. Raw microphone audio is not saved during normal use.

## Requirements

I developed the project on Windows with Python 3.11.9. The camera features need a working webcam; audio features need a microphone. CUDA is optional. Model downloads and public dataset setup require internet access.

I create the environment and install the dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some dependencies are unpinned, so a fresh installation may differ from the evaluated environment. See the final report for the recorded setup and performance conditions.

I use `tools/verify_clean_environment.py` to check the runtime packages. Its import map currently omits `matplotlib` and `reportlab`, so it marks those installed report dependencies as failed. I check their imports separately with `python -c "import matplotlib, reportlab"`.

## Download the models

I download weights rather than include them in this repository. I use the setup script to check each asset against its recorded size and SHA-256 checksum.

```powershell
python tools/setup_models.py --list
python tools/setup_models.py --download --only yolov8n --only yamnet
```

The model sources and licence notes are in [the model manifest](models/model_manifest.json) and [third-party inventory](docs/third_party_model_inventory.md). Whisper weights are downloaded on first use; `tools/prepare_whisper_model.py` can prepare them before a demonstration.

I check the camera before running the dashboard:

```powershell
python src/video/camera_test.py
python src/video/driver_visual_prototype.py
```

For optional audio:

```powershell
python tools/audio_model_smoke_test.py --validate-model
python tools/audio_model_smoke_test.py --list-devices
$env:CABINSPECTOR_USE_AUDIO = "1"
python src/video/driver_visual_prototype.py
```

If the microphone is unavailable, the visual loop can continue. The audio panel reports its status.

## Set up telemetry replay

I use the public Driving Events Dataset for replay, without personal trip recordings. I download the dataset and prepare the Random Forest comparison:

```powershell
python tools/download_driving_events_dataset.py
python tools/audit_driving_events_dataset.py
python tools/prepare_driving_events_dataset.py
python tools/train_telemetry_models.py --device cpu
python tools/export_telemetry_model.py
```

For PRIMUS, I download the verified checkpoint (approximately 622 MB), evaluate event-centred transfer, and export the encoder and fitted head:

```powershell
python tools/setup_models.py --download --only primus_checkpoint
python tools/evaluate_primus_telemetry_transfer.py
python tools/export_primus_telemetry_model.py
python tools/primus_event_model_smoke_test.py --trip 1 --event-index 0
```

I start the combined dashboard after setting up the models and data:

```powershell
$env:CABINSPECTOR_USE_AUDIO = "1"
$env:CABINSPECTOR_USE_TELEMETRY_REPLAY = "1"
$env:CABINSPECTOR_TELEMETRY_REPLAY_TRIP = "1"
$env:CABINSPECTOR_TELEMETRY_REPLAY_SPEED = "10"
$env:CABINSPECTOR_TELEMETRY_REPLAY_MODEL = "primus"
python src/video/driver_visual_prototype.py
```

I set the replay model to `random_forest` when I use the comparison model. This is timed replay of recorded events, not a connection to a live phone sensor. The exported heads use all annotated events for demonstration; held-out evaluation fits separate heads without the test trip.

## Dashboard controls

I use these keys while the dashboard window is focused.

| Key | Action |
|---|---|
| C | Calibrate the upper-body safe zone |
| V | Calibrate normal speaking volume for five seconds |
| T | Show or hide the temporary transcript |
| R | Start or stop transcript recording |
| E | Export recorded transcript text |
| P | Restart telemetry replay |
| S | Save a screenshot |
| D | Toggle debug overlays |
| Q / Esc | Close the application |

Transcript recording is off by default. Showing text does not turn on recording. Use recording and export controls only with the consent of people whose speech is captured. Automatic alert screenshots are enabled by default; disable them when images should not be retained:

```powershell
$env:CABINSPECTOR_ENABLE_AUTO_SCREENSHOTS = "0"
$env:CABINSPECTOR_STORE_TRANSCRIPTS = "0"
```

For a bounded run, set `$env:CABINSPECTOR_MAX_FRAMES = "900"`. Zero leaves the dashboard running until it is closed.

## Tests and evaluation

I run the automated suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

My revised report records 83 passing tests. They cover rules, privacy controls, replay, metrics, and other software behaviour; they do not measure real camera or speech-recognition accuracy.

I reproduce the synthetic checks and corrected continuous telemetry protocol with:

```powershell
python tools/evaluate_visual_scenarios.py
python tools/evaluate_audio_protocol.py
python tools/evaluate_fusion_sensitivity.py
python tools/evaluate_continuous_telemetry.py --stride-seconds 1.0
python tools/analyze_threshold_sensitivity.py
```

I fit each model on two trips and test the third in the continuous telemetry evaluator, including an always-NORMAL baseline. It reports false-positive windows using one-second decision exposure and separately groups alert episodes.

| Model at confidence gate 0.55 | Accuracy | Macro-F1 |
|---|---:|---:|
| PRIMUS with logistic head | 0.5628 | 0.4194 |
| Random Forest | 0.7103 | 0.4683 |
| Always NORMAL | 0.6558 | 0.1584 |

These results cover 3,728 overlapping windows from three trips. Prior model-family selection used the same trips, and the models use different training-window representations. This is exploratory evidence, not an independent deployment benchmark.

I keep the result files behind the figures in [docs/evaluation/results](docs/evaluation/results/README.md). They include the corrected public telemetry predictions, synthetic checks, and resource summaries.

For a short resource check that disables screenshots and transcript recording:

```powershell
python tools/measure_combined_resources.py --frames 900 --audio --telemetry
```

Real visual and audio evaluations require labelled, consent-cleared inputs. Follow [the evaluation protocols](docs/evaluation/evaluation_protocols.md); the blank [audio manifest](docs/evaluation/audio_manifest_template.csv) is a template, not a dataset. Keep filled manifests and recordings outside the repository.

## Project layout

I keep the application, tests, model setup, and documentation in separate folders. The [June workplan](docs/diagrams/workplan_timeline.mmd) records my earlier schedule; it is historical planning, not a claim that every planned activity was completed.

```text
src/                 Camera, audio, telemetry, and evaluation code
tools/               Setup, training, diagnostics, and evaluation scripts
tests/               Automated tests
models/              Download manifest and model attribution
data/audio/          Bilingual safety-rule configuration only
docs/diagrams/       Architecture and flow diagrams
docs/evaluation/     Evaluation protocols, blank manifest, and retained results
docs/report_revision/ Editable final report and figures
docs/report/         Final report PDF
```

I keep the design, methods, results and limitations in the [final report](docs/report/CabInspector_Final_Report.pdf). Its [Markdown source](docs/report_revision/CabInspector_Final_Report_Revised.md) and figures are included. I check its word limits with:

```powershell
python tools/verify_report_limits.py
```

I rebuild the PDF with [the report export command](docs/report_revision/BUILD_REPORT.md), which selects the manual camera image and its caption. On Windows, the builder uses `C:/Windows/Fonts/ARIALUNI.ttf` and `arialbd.ttf`.

I save exported PDFs under `output/`; logs and evaluation JSON go under `outputs/`. Both are excluded from Git. The only included file under `data/` is the configurable bilingual keyword list, `data/audio/safety_rules.json`; it contains no recordings or participant data. The report includes a manual screenshot with a usable camera image; it illustrates display and status output, not measured detection accuracy.

## Limitations

I have not yet validated the prototype for deployment.

The visual rules depend on lighting, camera position, visible landmarks, and object size. Asynchronous object detections can be combined with newer landmarks. Related phone cues can contribute more than once to the review score, whose weights are not calibrated against labelled risk outcomes.

Real visual precision/recall, ASR WER/CER, safety-category accuracy, and stakeholder usability have not been measured. Telemetry has a substantial false-positive burden and remains recorded replay. Short resource runs on one laptop do not establish long-session stability.

## Attribution

I use Ultralytics YOLO, Google MediaPipe and YAMNet, OpenAI Whisper, SYSTRAN Faster-Whisper, Nokia Bell Labs PRIMUS, and the public Driving Events Dataset. Their code, datasets, and weights retain their respective terms. See [the inventory](docs/third_party_model_inventory.md) and [PRIMUS provenance](models/telemetry/pretrained/primus/PROVENANCE.md). No project-wide licence has been selected.
