"""EPM / Y-maze / T-maze style derived metrics: arm entries and the
classic spontaneous-alternation percentage, both built from a zone
transitions table (tracking.location.calculate_transitions)."""


def calculate_arm_entries(transitions, roi_names):
    """Count how many times the animal entered each named zone -- the
    standard "arm entries" metric used in EPM/Y-maze/T-maze studies."""
    if transitions.empty:
        return {name: 0 for name in roi_names}
    counts = transitions["To_ROI"].value_counts()
    return {name: int(counts.get(name, 0)) for name in roi_names}


def calculate_alternation(transitions):
    """
    Classic spontaneous alternation %, as used for Y-maze/T-maze/radial-arm
    studies: slide a window of 3 consecutive zone entries across the full
    entry sequence; a triplet "alternates" if all 3 entries are different
    zones. Alternation % = alternating triplets / total triplets.
    Needs >= 3 entries; returns None for Alternation_percent otherwise
    (e.g. a 2-zone light-dark box can never produce a 3-different triplet).
    """
    sequence = transitions["To_ROI"].tolist() if not transitions.empty else []
    total_entries = len(sequence)

    if total_entries < 3:
        return {
            "Total_arm_entries": total_entries,
            "Alternating_triplets": 0,
            "Total_triplets": 0,
            "Alternation_percent": None,
        }

    triplets = [sequence[i:i + 3] for i in range(total_entries - 2)]
    alternating = sum(1 for t in triplets if len(set(t)) == 3)

    return {
        "Total_arm_entries": total_entries,
        "Alternating_triplets": alternating,
        "Total_triplets": len(triplets),
        "Alternation_percent": (alternating / len(triplets) * 100) if triplets else None,
    }


# -----------------------------
# Interaction bout extraction + manual tagging -- new in V6
# -----------------------------
