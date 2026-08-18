import json
import pandas as pd

RAW_PATH = "data/raw/arxiv-metadata-oai-snapshot.json"

CATEGORIES_TO_KEEP = {"cs.LG", "cs.CL"}
YEARS_TO_KEEP = {2021, 2024}
MIN_ABSTRACT_WORDS = 20


def paper_matches_filters(paper):
    # Check 1: category
    categories = set(paper.get("categories", "").split())
    if not categories.intersection(CATEGORIES_TO_KEEP):
        return False

    
    # Check 2: year 
    versions = paper.get("versions", [])
    if not versions:
        return False
    year = int(versions[0]["created"].split()[3])
    if year not in YEARS_TO_KEEP:
        return False

    # Check 3: abstract length
    abstract = paper.get("abstract", "") or ""
    if len(abstract.split()) < MIN_ABSTRACT_WORDS:
        return False

    return True


records = []

with open(RAW_PATH, "r") as f:
    for line in f:
        paper = json.loads(line)
        if paper_matches_filters(paper):
            year = int(paper["versions"][0]["created"].split()[3])
            records.append({
                "id": paper.get("id"),
                "title": paper.get("title", "").strip().replace("\n", " "),
                "abstract": paper.get("abstract", "").strip().replace("\n", " "),
                "categories": paper.get("categories", ""),
                "year": year,
            })

df = pd.DataFrame(records)

df.to_parquet("data/processed/arxiv_filtered.parquet", index=False)
print("Saved to data/processed/arxiv_filtered.parquet") 
