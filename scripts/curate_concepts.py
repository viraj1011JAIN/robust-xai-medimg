from src.concepts.curator import ConceptCurator

# TODO: set your actual CSV and image folder
CSV = "data/derm_meta.csv"
IMAGES = "data/derm_images"
OUT = "data/concepts/derm7pt"

cur = ConceptCurator(CSV, IMAGES, OUT)
cur.curate_clinical_concept("pigment_network", "pigment_network==1", "pigment_network==0")
cur.curate_clinical_concept("streaks", "streaks==1", "streaks==0")
cur.curate_clinical_concept("regression_structures", "regression_structures==1", "regression_structures==0")
cur.curate_clinical_concept("dots_globules", "dots_globules==1", "dots_globules==0")
cur.curate_clinical_concept("negative_network", "negative_network==1", "negative_network==0")
cur.curate_artifact_concept("artifacts_neg", "has_artifact")
print("Curated concepts to", OUT)
