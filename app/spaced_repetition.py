import datetime
from typing import Dict, Any, List


def calculate_sm2_interval(
    repetitions: int = 0,
    ease_factor: float = 2.5,
    previous_interval: int = 1,
    quality: int = 4
) -> Dict[str, Any]:
    """
    SuperMemo SM-2 Spaced Repetition Algorithm.
    Calculates next review interval in days based on student recall quality (0 to 5).
    """
    quality = max(0, min(5, quality))
    ease_factor = max(1.3, ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))

    if quality >= 3:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = max(1, int(round(previous_interval * ease_factor)))
        repetitions += 1
    else:
        repetitions = 0
        interval = 1

    next_review_date = (datetime.date.today() + datetime.timedelta(days=interval)).isoformat()

    return {
        "repetitions": repetitions,
        "ease_factor": round(ease_factor, 2),
        "interval_days": interval,
        "next_review_date": next_review_date
    }


def schedule_card_review(card: Dict[str, Any], rating: str = "good") -> Dict[str, Any]:
    """
    Shorthand reviewer for Flashcards with user-friendly ratings:
    'again' (quality=1), 'hard' (quality=3), 'good' (quality=4), 'easy' (quality=5).
    """
    quality_map = {
        "again": 1,
        "hard": 3,
        "good": 4,
        "easy": 5
    }
    q = quality_map.get(rating.lower(), 4)
    reps = card.get("repetitions", 0)
    ef = card.get("ease_factor", 2.5)
    prev_int = card.get("interval_days", 1)

    result = calculate_sm2_interval(repetitions=reps, ease_factor=ef, previous_interval=prev_int, quality=q)
    updated_card = dict(card)
    updated_card.update(result)
    updated_card["last_rating"] = rating.capitalize()
    return updated_card


def generate_deck_review_summary(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes learning status of a deck into stages:
    Learning (interval <= 1), Reviewing (interval <= 6), Mastered (interval > 6).
    """
    learning = 0
    reviewing = 0
    mastered = 0

    for c in cards:
        interval = c.get("interval_days", 1)
        if interval <= 1:
            learning += 1
        elif interval <= 6:
            reviewing += 1
        else:
            mastered += 1

    return {
        "total": len(cards),
        "learning": learning,
        "reviewing": reviewing,
        "mastered": mastered
    }
