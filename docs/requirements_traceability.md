# CabInspector Stakeholder and Requirements Traceability

I use this table for design analysis, not as a record of interviews. No stakeholder
interviews, participant study, or ethics-approved user trial was conducted for
this project. The requirements come from the project scope, the CM3020 AI 4.1 brief, and
the technical literature. They still need review by the intended users.

The report's literature review distinguishes inclusive design, design justice,
and radical inclusion rather than treating them as synonyms. Driver-monitoring
dataset engineering, Egyptian Arabic-English ASR, group-specific ASR/facial-
analysis evidence, human-automation levels, W3C accessibility guidance, and UK
ICO worker-monitoring guidance inform these provisional needs (see report
references [18]-[30]). These sources are not evidence that CabInspector has been
validated with their populations or deployment contexts.

| ID | Stakeholder / affected group | Need or concern | Requirement | Design / implementation evidence | Verification status |
|---|---|---|---|---|---|
| REQ-01 | Driver | Know when cabin monitoring is active and what an alert means | Show source, category, confidence/risk contribution, and reviewer-facing uncertainty | Dashboard panels, event categories, bounded risk breakdown, alert text | VERIFIED in UI smoke output; human-factors validation missing |
| REQ-02 | Driver | Avoid punitive or overconfident automated judgement | Alerts must support human review and must not be described as proof of misconduct, emotion, medical state, or legal fault | Risk score is heuristic; report and UI frame alerts as evidence | VERIFIED in design/code; policy acceptance missing |
| REQ-03 | Driver | Minimise privacy intrusion | Raw microphone audio is not saved by default; transcript persistence is opt-in; central event log contains no transcript text | `SpeechAnalysisPipeline`, `TranscriptStore`, `EVENT_LOG_FIELDS`, privacy tests | VERIFIED by tests and file audit |
| REQ-04 | Driver / passenger | Avoid silently monitoring speech | Make audio and transcript visibility/recording state explicit and controllable | Audio is opt-in; `T` and `R` controls; `LOG Off` default | VERIFIED in dashboard screenshot; consent/usability study missing |
| REQ-05 | Passenger | Not be misclassified as the driver | Treat microphone and camera evidence as uncertain about speaker/identity; do not infer passenger fault | Audio only supplies context; visual rules are driver-position assumptions; dashboard subtitles state `REVIEW ONLY` and `NO SPEAKER ID` | PARTIAL; passenger-specific camera tests and usability validation remain missing |
| REQ-06 | Fleet or ride-hailing operator | Review possible incidents efficiently | Provide time-stamped categories, source model, confidence, status, and bounded risk context | CSV event log, screenshots, telemetry replay status, risk breakdown | VERIFIED structurally; workflow/usability validation missing |
| REQ-07 | Human safety reviewer | Trace an alert to its evidence | Keep visual, audio, and telemetry fields separate and expose the reason for a fusion contribution | `risk_breakdown`, source-specific event fields, no hidden source labels in dashboard | VERIFIED by code/tests; reviewer study missing |
| REQ-08 | Human safety reviewer | Avoid false confidence from one weak signal | Use persistence/confidence gates and conservative cross-modal rules | Smoothing counters, audio smoother, telemetry confidence gate, speech-phone dependency | VERIFIED as code behaviour; threshold sensitivity is measured; calibration remains unvalidated |
| REQ-09 | Project assessor / supervisor | Reproduce the work | Provide environment, data provenance, setup commands, tests, evaluator commands, and known limitations | `requirements.txt`, `models/model_manifest.json`, checksum setup utility, provenance files, evaluation protocols, clean-environment report, automated test suite | VERIFIED locally; public publication and hardware-specific reproduction remain open |
| REQ-10 | Driver with Arabic, English, or mixed speech | Avoid assuming English-only interaction | Offer bilingual transcription configuration and report language-specific uncertainty | Faster-Whisper bilingual mode, Arabic normalisation and rendering tests | PARTIAL; formal consented WER/CER set missing |
| REQ-11 | Drivers with glasses, head coverings, different skin tones, facial hair, or different seating positions | Avoid presenting one camera configuration as universal | Test representative conditions and report coverage; allow calibration/fallback when landmarks fail | Safe-zone calibration, Haar fallback, Face Mesh, explicit visual limitations | PARTIAL; no representative labelled visual set |
| REQ-12 | Driver with hearing, speech, motor, or language differences | Avoid treating speech/loudness or hand posture as a universal proxy for risk | Keep audio/gesture signals optional, interpretable, and non-diagnostic; provide a way to review without them | Audio/telemetry opt-in, source-specific alerts, no medical claims | PARTIAL; accessibility review and alternative interaction study missing |
| REQ-13 | Data protection / ethics reviewer | Prevent inappropriate retention or secondary use | Document retention, consent, access, and deletion rules before collecting personal evaluation data | No raw audio in normal path; diagnostic artefacts explicitly identified; protocol document | PARTIAL; deployment governance and access control are outside prototype |
| REQ-14 | System operator | Keep the application responsive on available hardware | Measure bounded combined throughput, CPU, memory, GPU/VRAM, threads, stage timing, and clean shutdown; identify remaining latency/queue gaps | `measure_combined_resources.py`, instrumented three-run 900-frame summary, bounded frame mode, `nvidia-smi` samples | PARTIAL: visual/YOLO/YAMNet timing and observed audio/YOLO queue counters are verified; telemetry-startup, every internal queue, thermal data, and hardware replication remain open |
| REQ-15 | Project assessor | Demonstrate genuine multimodal orchestration | Integrate at least three pretrained models/data spaces and compare against meaningful baselines | YOLO/MediaPipe, YAMNet/Whisper, PRIMUS; trip-held-out telemetry RF comparison with different training-window representations; fusion sensitivity | PARTIAL; visual/audio real-world accuracy and live telemetry remain absent |

## Requirement interpretation

I distinguish implemented behaviour from deployment accuracy. In this prototype,
I check orchestration, privacy defaults, rule contracts, public-data telemetry
transfer, and bounded resource behaviour. It does not verify identity, intent, emotion, legal fault,
or universal performance across drivers and cabins.
