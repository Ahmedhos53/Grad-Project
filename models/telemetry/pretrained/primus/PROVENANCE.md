# PRIMUS checkpoint provenance

I keep the externally pretrained PRIMUS IMU encoder in this directory. It is deliberately separate from CabInspector's custom-trained Random Forest.

- **Model:** PRIMUS Model Checkpoint (`best_model.ckpt`)
- **Publisher record:** https://zenodo.org/records/15147513
- **Official source code:** https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders
- **Paper:** Das et al. (2025), *PRIMUS: Pretraining IMU Encoders with Multimodal Self-Supervision*, ICASSP 2025, https://arxiv.org/abs/2411.15127
- **Checkpoint licence:** CC-BY-4.0, as declared by Zenodo record 15147513.
- **Source-code licence:** BSD-3-Clause-Clear, as declared by the official repository.
- **Publisher file size:** 621,911,316 bytes.
- **Publisher checksum:** MD5 `badba8cbbd15cd685f2ed08ee7331aaa`.
- **Downloaded and MD5-verified on:** 18 August 2026.
- **Verified SHA-256:** `66392F3CEBF1DCBE079FC4B8BDF3B4460BB7BCAD8E47A217D114FCDF917FBA3F`.

The original checkpoint by itself is not a CabInspector driving-event classifier. On 18 August 2026, its frozen IMU encoder passed CabInspector's time-correct, leave-one-trip-out transfer evaluation with a separately fitted linear event head (mean macro-F1 0.7443; mean accuracy 0.8152). It is now accepted as the project's externally pretrained telemetry component, subject to the limitations in `event_model_metadata.json`.

Deployment-derived files in this directory:

- `imu_encoder_state.pt`: only the frozen IMU backbone extracted from the verified checkpoint; SHA-256 `28140EBF6BA054EB9BD47C149BAFA9DE11CCCEE56810675041580D9BD769E9B2`.
- `event_linear_head.joblib`: CabInspector classifier fitted on frozen embeddings from 169 public event-centred windows; SHA-256 `66BA2666CDA27A1A5AA038CAC0139CFF8E8CD6C98172F327D4A247E942F3DB68`.
- `event_model_metadata.json`: exact input contract, evaluation link, hashes, licences, and limitations.

The exported model is operational in standalone inference and is connected to the opt-in public-trip dashboard replay. This remains a recorded replay, not continuous live phone telemetry. Live telemetry would require a sensor adapter, sliding-window training, and live false-alarm evaluation.
