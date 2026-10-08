from app.spaced_repetition import calculate_sm2_interval, schedule_card_review, generate_deck_review_summary

def test_calculate_sm2_interval():
    # First successful recall
    res1 = calculate_sm2_interval(repetitions=0, ease_factor=2.5, previous_interval=1, quality=4)
    assert res1["repetitions"] == 1
    assert res1["interval_days"] == 1

    # Second successful recall
    res2 = calculate_sm2_interval(repetitions=1, ease_factor=res1["ease_factor"], previous_interval=res1["interval_days"], quality=4)
    assert res2["repetitions"] == 2
    assert res2["interval_days"] == 6

    # Third successful recall
    res3 = calculate_sm2_interval(repetitions=2, ease_factor=res2["ease_factor"], previous_interval=res2["interval_days"], quality=5)
    assert res3["repetitions"] == 3
    assert res3["interval_days"] > 6

    # Failed recall (reset)
    res_fail = calculate_sm2_interval(repetitions=3, ease_factor=res3["ease_factor"], previous_interval=res3["interval_days"], quality=1)
    assert res_fail["repetitions"] == 0
    assert res_fail["interval_days"] == 1
    print("[PASS] calculate_sm2_interval verified across repetition progressions")

def test_schedule_card_review():
    card = {"front": "What is PCB?", "back": "Process Control Block"}
    updated = schedule_card_review(card, rating="good")
    assert "interval_days" in updated
    assert "next_review_date" in updated
    assert updated["last_rating"] == "Good"
    print("[PASS] schedule_card_review verified")

def test_generate_deck_review_summary():
    cards = [
        {"front": "C1", "interval_days": 1},
        {"front": "C2", "interval_days": 5},
        {"front": "C3", "interval_days": 15}
    ]
    summary = generate_deck_review_summary(cards)
    assert summary["total"] == 3
    assert summary["learning"] == 1
    assert summary["reviewing"] == 1
    assert summary["mastered"] == 1
    print("[PASS] generate_deck_review_summary verified")

def main():
    print("=== Running Spaced Repetition Unit Tests ===")
    test_calculate_sm2_interval()
    test_schedule_card_review()
    test_generate_deck_review_summary()
    print("=== All Commit 24 Tests Passed! ===")

if __name__ == "__main__":
    main()
