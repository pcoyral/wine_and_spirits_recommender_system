"""The recommenders, and the one list of models compared everywhere."""


def build_models(cfg):
    """Every model in the comparison, in table order: the three baselines,
    Sentence-BERT ("semantic"), then any extra embedding models from
    config.yaml. Used by the demo, `make eval` and the simulated study, so
    they always compare the same models."""
    from sensoryrec.recommenders import attribute, semantic, tfidf
    from sensoryrec.recommenders.popularity import PopularityRecommender
    return [PopularityRecommender(), tfidf.from_config(cfg), attribute.from_config(cfg),
            semantic.from_config(cfg), *semantic.extra_from_config(cfg)]


def ild_vectors(models):
    """wine_id -> Sentence-BERT embedding, the one yardstick used to measure
    intra-list diversity for every model (DECISIONS.md, 21 Oct 2026)."""
    return next(m for m in models if m.name == "semantic").vectors()
