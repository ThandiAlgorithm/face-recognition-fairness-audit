# Harmonized Dataset Website v2 — ALL DATA

This version is designed to contain **all generated harmonized-dataset outputs** and a searchable 95,258-record manifest.

## Put the folder here

`face-recognition-fairness-audit/results/harmonized_dataset/final_dataset_website/`

## Copy all project data into the site

Open PowerShell in that folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup_all_data.ps1
```

The script copies everything currently under `results/harmonized_dataset/` (except the website itself) into:

`data/project_outputs/`

This includes the final manifest, seven label CSVs, dataset card, schema, summaries, previews, label inventory, taxonomy experiments, balancing scenarios, diversity audits, population pack and frozen-manifest metadata.

## Open the website correctly

Do **not** double-click `index.html` for the full data explorer. Browsers often block CSV loading from `file://` pages.

In the website folder run:

```powershell
python -m http.server 8000
```

Then open:

`http://localhost:8000`

Press `Ctrl+C` in PowerShell when you want to stop the local website server.

## Source images

The website does not duplicate all 95,258 face image files. It contains the complete metadata/manifests and preview sheets. The canonical manifest preserves each record's source image path and provenance.
