import numpy as np
import pandas as pd
from sentence_transformers import util
import logging

logging.basicConfig(level=logging.INFO)

def classify_news_expanded(news_text, taxonomy, model):
    news_vec = model.encode(news_text, convert_to_tensor=True)
    sector_scores = {}
    
    total_subindustries_processed = 0

    for sector, sub_industries in taxonomy.items():
        sub_texts = []

        for sub_name, data in sub_industries.items():
            # Omitir claves con metadatos
            if sub_name.startswith("_") or not isinstance(data, dict):
                continue

            desc = data.get("description", "")
            keywords = ", ".join(data.get("keywords", []))
            formatted_text = f"Sub-industry: {sub_name}. Description: {desc}. Keywords: {keywords}."
            
            sub_texts.append(formatted_text)

        if not sub_texts:
            continue

        total_subindustries_processed += len(sub_texts)

        sub_vecs = model.encode(sub_texts, convert_to_tensor=True)
        sims = util.cos_sim(news_vec, sub_vecs).cpu().numpy()[0]
        sector_scores[sector] = np.mean(sims)

    logging.info(f"Total sectores procesados: {len(sector_scores)}")
    logging.info(f"Total sub-industrias procesadas: {total_subindustries_processed}")

    # Normalización Softmax
    sectors = list(sector_scores.keys())
    scores = np.array(list(sector_scores.values()))
    temperature = 0.03
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / exp_scores.sum()

    return pd.Series(probs, index=sectors).sort_values(ascending=False)