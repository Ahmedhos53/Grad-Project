# CabInspector Evaluation Protocols

I run the automated checks locally, without retaining camera frames, raw
microphone samples, transcript text, or private telemetry. I interpret each
metric alongside the data and protocol that produced it.

## 1. Visual decision-rule contract

I run:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_visual_scenarios.py
```

The evaluator uses synthetic landmark/object-box scenarios for normal input,
phone-near-ear with hand contact, phone-near-ear without hand contact, drinking,
non-drinking, safe-zone containment, and safe-zone violation. It calls the same
decision functions used by the dashboard with model loading disabled. It writes
only scenario IDs, boolean outcomes, and aggregate counts to
`outputs/evaluation/visual_logic_evaluation.json`.

This is a rule-contract test, not a visual-model accuracy benchmark. A genuine
precision/recall study requires a separately labelled and consent-cleared set
of camera frames or clips with lighting, glasses, distance, occlusion, and
driver/non-driver cases represented.

## 2. Prompted visual evaluation (run only with consent)

I use this command for a local prompted run:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_visual_capture.py --consent-confirmed --trials 20 --frames-per-trial 20 --camera-index 0
```

The tool asks the operator to type `CONSENTED`, selects one behaviour per trial,
records the operator-entered expected label and condition tags, and processes
live camera frames in memory. It covers eye closure, safe-zone movement, phone
presence, phone-call context, and drinking context; each can be tested as
present or absent. Condition tags cover bright/normal/low light, glasses,
distance, head angle, and partial occlusion. The default safe box is 20%-80% of
frame width and 20%-95% of frame height; change it with `--safe-box-ratios` if
the agreed protocol uses a different calibration.

Run only while parked, with props rather than unsafe real actions, and only if
everyone visible has consented (and any applicable supervisor/ethics approval is
already in place). The tool disables audio, telemetry, and automatic
screenshots; it never writes frames or video. Results contain trial labels,
predictions, condition codes, timestamps, per-frame latency, and available
object-detector scores only. Rule decisions have no calibrated probability
confidence. Metrics include precision, recall, F1, false positives, false
negatives, and latency per behaviour. This is local engineering evidence, not a
representative or ethics-approved user study. The report shows missing
behaviours, absent positive/negative labels, and untested condition values so a
small or one-sided run cannot be mistaken for full coverage.

## 3. Audio contract and consented ASR protocol

I run the non-recording contract check:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_audio_protocol.py
```

It uses in-memory synthetic waveforms to check RMS/dBFS quality, clipping,
energy-gated segmentation, calibrated loudness levels, YAMNet category mapping,
and local bilingual safety-rule behaviour. It writes fixture IDs and categories
only; the controlled text and waveform samples are not written.

## 4. Consented audio-manifest evaluation

For a formal local-file evaluation, I collect short recordings with consent
and any applicable ethics/supervisor approval. I keep the audio and manifest
local and use unidentifying sample IDs. The evaluator accepts
uncompressed mono 16 kHz 16-bit PCM WAV files (maximum 30 seconds per clip and
100 rows). The CSV columns are:

```text
sample_id,audio_path,language,reference_text,expected_safety_categories,expected_loudness_level,microphone_type,noise_condition
```

I keep a header-only starter at
`docs/evaluation/audio_manifest_template.csv`. Copy it to a private local path
such as `data/audio/evaluation/manifest.csv` before adding anonymised sample IDs
and relative WAV paths. Do not commit a filled manifest or personal audio files.

`audio_path` is relative to the manifest folder. `language` is `ar`, `en`, or
`mixed`; `microphone_type` is `laptop` or `headset`. Safety categories are
optional and separated by `|` (for example, `PROFANITY|THREAT`); leave them
blank for safe speech. Loudness labels are optional and must be `QUIET`, `LOW`,
`NORMAL`, `HIGH`, or `VERY_HIGH`. If I include any, I also include a separate
consented normal-speech calibration WAV in the same folder. `noise_condition`
must be `quiet`, `road_noise`, `cabin_noise`, `music`, or `other`; include both
quiet and realistic background-noise conditions where consent and setup allow.
Use difficult safe negatives alongside the configured unsafe categories.

Run the evaluator only after the manifest and clips have been approved for this
purpose:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_consented_audio_manifest.py --manifest data\audio\evaluation\manifest.csv --calibration-wav normal_calibration.wav --consent-confirmed --output outputs\evaluation\consented_audio_evaluation.json
```

The tool asks for a second typed `CONSENTED` confirmation. It reads each WAV in
memory and does not modify or copy it. Its aggregate report contains WER/CER by
language and microphone type, language-detection agreement, local-rule safety
precision/recall/F1, loudness agreement/confusion counts, clipping/segmentation
counts, and transcription/pipeline latency. It never writes sample IDs, audio
paths, reference text, or transcript text. The optional `--safety-rules` points
to the same local rule file used by the application. I score the safety categories
from the transcribed text against the manifest labels, so transcription errors
also affect these measures. The report flags missing language or microphone conditions, absent background
noise, safety categories without positive examples, and missing loudness labels.

An output path is optional; without `--output` only aggregate JSON is printed.
Keep source WAVs and the reference manifest outside the public source archive.
Do not publish unsafe-language examples or individual transcripts.

## 5. Continuous public telemetry

I inspect the deployment-matched schedule without loading a model:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_continuous_telemetry.py --plan-only
```

I use a bounded smoke evaluation:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_continuous_telemetry.py --max-windows 3
```

The evaluator reads only the public Driving Events Dataset, resamples fixed
five-second/200 Hz windows at a configurable stride, assigns offline labels by
annotation overlap, and measures confusion, confidence, inference latency, and
false-positive windows using the configured stride. At a one-second stride,
the five-second windows overlap. It never feeds labels to the dashboard,
never reads `data/Telemetry/Recordings/`, and never stores sensor arrays in the
report. The result is explicitly an offline public-data evaluation; it is not a
live phone telemetry test.

## 6. Combined resource and stability measurement

I run a bounded local measurement with automatic screenshot persistence disabled:

```powershell
.\.venv\Scripts\python.exe tools\measure_combined_resources.py --frames 30 --audio --telemetry
```

The runner starts the existing dashboard in a subprocess, enables the bounded
frame limit, samples aggregate CPU/RSS/thread counts, and records whitelisted
lifecycle markers. It forces transcript recording off and automatic alert
screenshots off. It reports exit status, wall time, resource summaries, and
whether a traceback occurred. It does not save camera/audio media. Repeat the
measurement on the target machine for longer windows before making claims about
thermal behaviour, long-duration stability, or deployment readiness.

## 7. Manual acceptance actions requiring consent

I keep the following actions separate from the default automated
commands:

1. Confirm that the camera view is well lit and contains only the intended
   driver/test subject; do not record or publish frames without consent.
2. Run the dashboard with audio enabled for a short quiet-input check. Do not
   enable transcript recording unless the speaker has explicitly consented.
3. If collecting ASR results, use the consented local WAV manifest evaluator;
   keep reference text outside the central event log and public archive.
4. Exercise `C`, `V`, `T`, `R`, `E`, `P`, screenshot, and quit controls only as
   required by the approved checklist. Treat alerts as evidence for human review,
   not proof of misconduct, aggression, medical state, or legal fault.
