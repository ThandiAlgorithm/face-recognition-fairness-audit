# Measurement validity limitations (auto-generated draft, edit wording before use)

Skin tone was measured once per aligned, non-augmented crop with the frozen ROI in preprocessing.yaml. Status counts: {'ok': 128709, 'greyscale_no_chroma': 4738, 'invalid_skin_roi': 1334, 'ita_undefined': 204, 'not_eligible_measurement_audit': 54}.

1. There is no ground-truth skin-tone oracle in LFW, FairFace or UTKFace. The reliability tables report agreement between measurement variants (patches, median vs mean, Phase 0 vs MEA). Agreement shows the measurement is stable. It does not show the measurement is accurate.
2. ITA depends on scene lighting, white balance and camera processing, not only on skin. Check the VIS brightness results before reading group differences as skin differences.
3. ROI failure is not random if MEA_roi_failure_tests.csv shows differences by group. Invalid ROI, greyscale and undefined ITA images are flagged, not deleted.
4. Within 5 degrees of the 28 degree threshold, group membership is unstable. See MEA_boundary_sensitivity.csv.
5. FairFace images are publisher crops with padding. The same fixed ROI covers a different face area than on project-aligned LFW and UTKFace crops.
6. Albinism and other visible phenotypes are not inferred from real photographs. Scars, acne, facial difference and disability are not labelled. No manual review of real images was performed.
