from app.quiz_evaluator import evaluate_quiz_submission

def test_evaluate_quiz_submission_perfect():
    quiz = [
        {"question": "What is a process?", "answer": "Program in execution", "explanation": "See page 4."}
    ]
    answers = {1: "Program in execution"}
    res = evaluate_quiz_submission(quiz, answers)
    assert res["score"] == 1
    assert res["percentage"] == 100.0
    assert "Mastery" in res["mastery_level"]
    assert len(res["revision_needed"]) == 0
    print("[PASS] Perfect quiz score evaluated")

def test_evaluate_quiz_submission_partial():
    quiz = [
        {"question": "Q1", "answer": "A", "explanation": "Topic A"},
        {"question": "Q2", "answer": "B", "explanation": "Topic B"}
    ]
    answers = {1: "A", 2: "Wrong"}
    res = evaluate_quiz_submission(quiz, answers)
    assert res["score"] == 1
    assert res["percentage"] == 50.0
    assert len(res["revision_needed"]) == 1
    assert res["revision_needed"][0]["question_num"] == 2
    print("[PASS] Partial quiz score and revision recommendation verified")

def main():
    print("=== Running Quiz Evaluator Unit Tests ===")
    test_evaluate_quiz_submission_perfect()
    test_evaluate_quiz_submission_partial()
    print("=== All Commit 16 Tests Passed! ===")

if __name__ == "__main__":
    main()
