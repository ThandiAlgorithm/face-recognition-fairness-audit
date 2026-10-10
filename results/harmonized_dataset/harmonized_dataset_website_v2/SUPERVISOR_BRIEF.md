# Supervisor / Partner Brief

- Source inventory: 135,039 records across FairFace, UTKFace and LFW.
- Final frozen product: 95,258 valid records across seven explicit labels.
- Harmonization was conservative: source labels and provenance were preserved; broad Asian records were not visually subdivided.
- Strict parity was tested but rejected because it would retain only 26.39% (all trainable labels) or 54.04% (seven explicit labels) of the original pool.
- A volume-aware candidate retained 96,432 records before a soft 35/65 gender floor.
- The gender floor affected only Middle Eastern records and removed 1,160 male-labelled images.
- Exact ITA parity was rejected because it would destroy substantial within-group volume; ITA is reported as a separate audit variable.
- Four labels (East Asian, Southeast Asian, Middle Eastern, Latino) are entirely sourced from FairFace in the final product. This is the main provenance limitation.
- The dataset should be described as harmonized, diversity-aware and representation-balanced, not perfectly fair or universally representative.
- Model training is a separate workstream; the dataset manifest and documentation are the final product of this workstream.
