def calculate_similarity(f1, f2):
    if not f1 or not f2:
        return 0.0

    # confidence similarity
    conf_sim = 1 - abs(f1["cow_confidence"] - f2["cow_confidence"])

    # class match bonus
    class_bonus = 0.2 if f1["class"] == f2["class"] else 0.0

    score = (0.8 * conf_sim) + class_bonus

    # clamp
    if score < 0:
        score = 0
    if score > 1:
        score = 1

    return score
