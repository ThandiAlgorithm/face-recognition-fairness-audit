# Dataset-shift interpretation limits (auto-generated draft, edit wording before use)

Images in the shift analysis set: {'fairface': 93151, 'lfw': 12878, 'utkface': 21605}.

1. Only these axes are compared across datasets: ['skin_group_ita', 'ita_group', 'gender', 'age_band', 'skin_ITA_deg', 'skin_L_star', 'skin_hue_deg', 'brightness_L_mean', 'contrast_L_sd', 'sharpness_lapvar']. Native race labels (['race']) use different schemes in each dataset and are only analysed within a dataset.
2. Gender: Binary native labels (FairFace, UTKFace) normalised to male/female; LFW gender only from a documented external label file, otherwise unknown and excluded from gender comparisons.
3. Age: FairFace native bands and UTKFace ages bucketed into the same categories.yaml bands; LFW has no age labels.
4. Skin tone and visual measures come from one pixel pipeline, but the images differ in capture, cropping and compression. FairFace crops are publisher-made with padding, LFW and UTKFace crops are project-aligned. A difference in a measure can come from the image source, not only from the people shown.
5. Nominal balance (label counts) and effective balance (image conditions per group) are reported separately. No composite fairness score is created.
6. With large n, small differences reach significance. Read effect sizes (Cliff's delta, Cramér's V, Wasserstein distance) with their intervals, not p-values alone.
