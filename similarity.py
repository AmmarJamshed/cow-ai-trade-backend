def calculate_similarity(f1, f2):
    """
    Returns a similarity score BETWEEN 0.0 and 1.0
    """

    if not isinstance(f1, dict) or not isinstance(f2, dict):
        return 0.0

    # ------------------------
    # 1. Detection confidence similarity
    # ------------------------
    c1 = f1.get("cow_confidence", 0.0)
    c2 = f2.get("cow_confidence", 0.0)

    # Clamp confidence values
    c1 = max(0.0, min(float(c1), 1.0))
    c2 = max(0.0, min(float(c2), 1.0))

    conf_sim = 1.0 - abs(c1 - c2)  # always 0–1
    conf_sim = max(0.0, min(conf_sim, 1.0))

    # ------------------------
    # 2. Bounding box similarity
    # ------------------------
    b1 = f1.get("bbox", [0, 0, 0, 0])
    b2 = f2.get("bbox", [0, 0, 0, 0])

    if (
        not isinstance(b1, (list, tuple)) or
        not isinstance(b2, (list, tuple)) or
        len(b1) != 4 or
        len(b2) != 4
    ):
        bbox_sim = 0.5
    else:
        try:
            diff = sum(abs(float(a) - float(b)) for a, b in zip(b1, b2))
            bbox_sim = 1.0 / (1.0 + diff)   # smoothly decays
            bbox_sim = max(0.0, min(bbox_sim, 1.0))
        except Exception:
            bbox_sim = 0.5

    # ------------------------
    # 3. Disease bonus (small, capped)
    # ------------------------
    disease_bonus = 0.0
    d1 = f1.get("disease")
    d2 = f2.get("disease")

    if d1 and d2 and d1 == d2:
        disease_bonus = 0.1  # MAX bonus

    # ------------------------
    # 4. Weighted final score
    # ------------------------
    score = (0.6 * conf_sim) + (0.3 * bbox_sim) + disease_bonus

    # HARD SAFETY CLAMP
    score = max(0.0, min(score, 1.0))

    return round(score, 4)
