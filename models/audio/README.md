# YAMNet model asset

I use the official metadata-enabled YAMNet audio-classification model,
`yamnet_audio_classifier_with_metadata.tflite`, in CabInspector.

- Source: https://storage.googleapis.com/mediapipe-assets/yamnet_audio_classifier_with_metadata.tflite?generation=1661875980774466
- SHA-256: `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`
- Labels: 521 AudioSet categories, packaged in the model metadata.
- Runtime: Google LiteRT through `ai-edge-litert`.

I keep raw microphone audio out of storage during normal use. The runtime keeps only the
latest model result in memory for dashboard, logging, and evidence fusion.
