LOW_CONFIDENCE=0.3
MED_CONFIDENCE=0.44
HIGH_CONFIDENCE=0.75

def get_confidence_feedback(confidence: float):
    """Returns confidence level str and needs_review bool, and review_reason str; 
    based on confidence threshold global constants"""
    needs_review = False
    review_reason = None
    if confidence > HIGH_CONFIDENCE:
        confidence_rank="high"
    elif confidence > MED_CONFIDENCE:
        confidence_rank="medium"
    elif confidence > LOW_CONFIDENCE:
        confidence_rank="low"
        needs_review = True
        review_reason = "low_confidence"
    else:
        confidence_rank = "very_low"
        needs_review = True
        review_reason = "very_low_confidence"

    return confidence_rank, needs_review, review_reason