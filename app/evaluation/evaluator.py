
class RAGEvaluator:

    def evaluate_answer(self, answer, expected_keywords):
        answer_lower = answer.lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]

        score = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords else 0.0
        )

        return {
            "score": score,
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                keyword
                for keyword in expected_keywords
                if keyword not in matched_keywords
            ],
            "passed": score == 1.0
        }

    def evaluate_abstention(self, answer):
        expected_phrase = (
            "do not contain enough information"
        )

        return {
            "passed": expected_phrase in answer.lower()
        }
