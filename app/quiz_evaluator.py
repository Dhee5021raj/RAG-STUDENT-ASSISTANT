from typing import List, Dict, Any


def evaluate_quiz_submission(
    quiz_questions: List[Dict[str, Any]],
    user_answers: Dict[int, str]
) -> Dict[str, Any]:
    """
    Evaluates a completed quiz submission across all questions.
    Computes overall score, percentage, mastery rating, and compiles
    targeted revision recommendations for incorrect questions.
    """
    if not quiz_questions:
        return {
            "score": 0,
            "total": 0,
            "percentage": 0.0,
            "mastery_level": "No Assessment",
            "breakdown": [],
            "revision_needed": []
        }

    score = 0
    total = len(quiz_questions)
    breakdown = []
    revision_needed = []

    for idx, q in enumerate(quiz_questions, 1):
        correct_ans = q.get("answer", "").strip()
        user_ans = user_answers.get(idx, "").strip()
        is_correct = (user_ans == correct_ans) and (bool(user_ans))

        if is_correct:
            score += 1
        else:
            revision_needed.append({
                "question_num": idx,
                "question": q.get("question", ""),
                "expected": correct_ans,
                "explanation": q.get("explanation", ""),
                "study_recommendation": f"Review explanation: {q.get('explanation', 'Consult source material.')}"
            })

        breakdown.append({
            "question_num": idx,
            "question": q.get("question", ""),
            "user_answer": user_ans or "Unanswered",
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })

    pct = round((score / total) * 100.0, 1)

    if pct >= 90:
        mastery = "🏆 Mastery (A+)"
    elif pct >= 75:
        mastery = "⭐ Proficient (B/A)"
    elif pct >= 50:
        mastery = "📘 Developing (C)"
    else:
        mastery = "⚠️ Needs Revision"

    return {
        "score": score,
        "total": total,
        "percentage": pct,
        "mastery_level": mastery,
        "breakdown": breakdown,
        "revision_needed": revision_needed
    }
