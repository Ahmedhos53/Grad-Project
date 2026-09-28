# Third-Party Model Sources and Licence Review

I record the upstream sources and available licence evidence for
the current CabInspector model paths. I keep this as a provenance record.
Before redistributing weights or publishing the application, I check the exact
upstream files and their terms.

| Component | Upstream/source | Licence evidence recorded | Runtime/use in CabInspector | Publication note |
|---|---|---|---|---|
| YOLOv8n object weights and Ultralytics package | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Ultralytics licensing](https://www.ultralytics.com/license) | Ultralytics currently states AGPL-3.0 is its open-source option and that YOLO trained models are covered by AGPL-3.0; enterprise terms are offered for other uses. | PyTorch/Ultralytics; periodic phone/container detection | Application distribution requires review of the AGPL-3.0 obligations. Weight binaries are omitted from the repository and fetched through the checksum manifest. |
| MediaPipe Face Mesh, Pose, and Hands | [Google AI Edge MediaPipe](https://github.com/google-ai-edge/mediapipe) and papers [2], [3] in the report | MediaPipe repository code is Apache-2.0. Bundled model assets can have separate terms; their individual terms are not itemised here. | Python MediaPipe; live landmarks | Keep upstream notices and verify the exact bundled assets if redistributing them. |
| YAMNet | [TensorFlow Models YAMNet implementation](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet); deployed TFLite source and SHA-256 are in `models/model_manifest.json` | The TensorFlow Models implementation identifies Apache-2.0. The exact licence for the MediaPipe-hosted TFLite file in this repository was not independently verified. | LiteRT; 521 AudioSet categories collapsed to project context labels | Do not assume the implementation licence automatically covers the separately hosted TFLite file; verify before redistributing that asset. |
| Whisper large-v3-turbo / Faster-Whisper | [OpenAI Whisper](https://github.com/openai/whisper) and [SYSTRAN Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) | OpenAI states Whisper code and weights are MIT; Faster-Whisper declares MIT. | CTranslate2; in-memory bilingual utterance transcription | The application uses local inference. Separately downloaded model cache files are not included in the repository. |
| PRIMUS encoder checkpoint and inference implementation | [Nokia Bell Labs PRIMUS repository](https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders), [Zenodo record 15147513](https://zenodo.org/records/15147513), and [PROVENANCE.md](../models/telemetry/pretrained/primus/PROVENANCE.md) | Local provenance records CC BY-4.0 for the checkpoint and BSD-3-Clause-Clear for the source code. Check the Zenodo record and retain attribution. | PyTorch frozen encoder plus project-fitted linear event head; public replay only | The large checkpoint and derived weight binaries are omitted from the repository; the setup manifest verifies the source checkpoint hash. |

## My checks before public release

Before distributing the application, I review the Ultralytics AGPL-3.0 implications
for the full application and confirm the exact redistribution terms for the
MediaPipe-hosted YAMNet TFLite asset and bundled MediaPipe models. A project-wide licence has not been selected.
