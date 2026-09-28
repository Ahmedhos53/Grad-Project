# CabInspector Inclusion and Radical-Inclusion Audit

I use this audit to examine the design before deployment. It is not a fairness result: the project
has no representative participant dataset, no demographic labels, and no
stakeholder interviews. I use it to identify who my current assumptions could exclude and the evidence
I still need before making stronger claims.

## Working definitions

I use Inclusive Design Research Centre's definition of inclusive design as
considering the full range of human diversity, including ability, language,
culture, gender, age, and other human differences
([IDRC](https://idrc.ocadu.ca/)). I use design justice to mean that design
processes should be led by and accountable to people affected by their outcomes
([Costanza-Chock](https://designjustice.mitpress.mit.edu/)). I use radical
inclusion narrowly to consider how I could identify and remove structural barriers; the
Sierra Leone source is an education policy, so I do not transfer its specific
policy conclusions to a driver-monitoring system
([National Policy on Radical Inclusion in Schools](https://mbsse.gov.sl/wp-content/uploads/2021/04/Radical-Inclusion-Policy.pdf)).
CabInspector has not undertaken a participatory or radical-inclusion process; the
terms guide questions for future work rather than describe an achieved outcome.

| Area | Current assumption or risk | Potentially excluded / burdened group | Current mitigation | Evidence still required |
|---|---|---|---|---|
| Language | Whisper and safety rules are configured for Arabic/English, but dialect and code-switching errors remain possible | Egyptian Arabic speakers, code-switching speakers, non-Arabic/English speakers | Bilingual mode, Arabic normalisation, configurable rules, transcript uncertainty | Consent-cleared WER/CER by language/dialect and microphone condition |
| Colour and text status cues | Colour badges can be harder to interpret for users with colour-vision differences, low vision, or monochrome displays | Users with colour-vision deficiency or other visual-access needs | Status rows pair text labels and values with colour; the Risk Summary shows a textual LOW/MODERATE/HIGH label and all five source contributions in compact text form | Contrast/scale review and user testing; no WCAG conformance or accessibility-study claim |
| Hearing and speech | Loudness and speech activity are treated as useful context | Deaf/hard-of-hearing drivers, non-speaking drivers, speech disabilities | Audio is opt-in and source-specific; audio is not the only route to an alert | Accessibility review showing the dashboard does not make audio participation mandatory |
| Vision and face landmarks | Face/eye/pose/hand landmarks can fail with glasses, head coverings, facial hair, lighting, camera angle, or occlusion | Drivers with visual aids, religious/cultural head coverings, different facial features, or atypical seating | Haar fallback, Face Mesh repair, safe-zone calibration, explicit “no face” state | Labelled visual tests across lighting, occlusion, glasses, camera distance, and seating positions |
| Skin tone and appearance | Pretrained visual models may have uneven performance across appearances | Drivers with under-represented skin tones or appearance characteristics | No fairness claim is made; outputs are review evidence only | Stratified labelled evaluation with a documented, ethically justified protocol |
| Body movement | Hand/arm geometry is used as a safe-zone and contact signal | Drivers with motor differences, limb differences, restricted movement, or adaptive vehicle controls | Rules are interpretable and optional; no medical inference | Scenario review with accessibility specialists and false-alert analysis |
| Passenger privacy | Cabin microphone/camera can capture passengers even though the target is the driver | Passengers, children, visitors, or bystanders | No identity inference; raw audio not saved by default; transcript logging opt-in | Consent/signage, masking policy, retention/deletion controls, and passenger-specific test cases |
| Power and hardware | Combined run uses substantial memory and high peak aggregate CPU | Users on lower-spec laptops or battery-powered devices | Fast-demo settings, optional modalities, bounded resource evaluator | Replicated resource measurements across target hardware and longer sessions |
| Interpretation and language | “Risk” can be understood as guilt or punishment | Drivers, operators, reviewers, and communities subject to automated escalation | UI/report language says human review and evidence, not proof | Stakeholder review of labels, escalation policy, and alert wording |
| Data governance | Local files can still be copied, opened, or retained outside the app | All occupants whose data enters evaluation | Explicit diagnostic artefact inventory and no normal raw-audio persistence | Approved retention schedule, access controls, deletion process, and ethics approval where required |

## Actions before deployment claims

I would complete these checks before drawing deployment conclusions.

1. Obtain supervisor/ethics guidance before collecting new personal audio or video.
2. Validate alert wording and consent flow with intended reviewers and drivers,
   and document the feedback received.
3. Build a representative, consent-cleared visual/audio evaluation set with
   inclusion dimensions chosen in advance and only the minimum metadata needed.
4. Report subgroup coverage and failure rates, not only aggregate accuracy.
5. Keep the current prototype boundary: recorded public telemetry replay, local
   review support, and no automatic disciplinary or medical decision.
