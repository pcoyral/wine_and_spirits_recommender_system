"""`make human-cards`: the 30 test wines for the hand-written queries (D1, D6).

The wines are drawn from the 200 test wines, spread over the variety bands
like the test set (at least `min_per_band` per band, the rest proportional),
with the fixed seed. The cards are shuffled so a participant does not see
the bands in order.

Outputs
- data/human_cards.csv            card, wine_id, note: what the researcher shows
- data/human_queries_template.csv one row per card, for the answers
  (copy it to data/human_queries.csv and fill it in; see
  docs/human_query_protocol.md)
"""
import pandas as pd

from sensoryrec.config import load_config, repo_path
from sensoryrec.data import stratified_sample

TEMPLATE_COLUMNS = ["participant_id", "card", "wine_id", "query", "self_rated_knowledge", "consent", "date"]


def select_human_wines(test_ids, n, minimum, seed):
    """n test wines stratified by band (test_ids has wine_id and band),
    in a shuffled order."""
    chosen = stratified_sample(test_ids, n, minimum, seed)
    return chosen.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    cfg = load_config()
    h = cfg["human"]
    processed = repo_path(cfg["paths"]["processed_dir"])
    wines = pd.read_parquet(processed / "wines.parquet").set_index("wine_id")
    test_ids = pd.read_csv(processed / "test_ids.csv")

    chosen = select_human_wines(test_ids, h["n_wines"], h["min_per_band"], cfg["seed"])
    cards = pd.DataFrame({"card": range(1, len(chosen) + 1), "wine_id": chosen["wine_id"],
                          "note": wines.loc[chosen["wine_id"], "description"].to_numpy()})
    cards.to_csv(repo_path(h["cards"]), index=False)

    template = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    template["card"], template["wine_id"] = cards["card"], cards["wine_id"]
    template.to_csv(repo_path(h["template"]), index=False)
    print(f"{len(cards)} cards; by band: {chosen['band'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
