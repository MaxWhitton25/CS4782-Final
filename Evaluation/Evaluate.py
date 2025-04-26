import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

# Internal variable to hold the ROUGE evaluator
_rouge = None

def _setup_metrics():
    """
    Internal setup function to initialize required resources.
    Called automatically on first use of evaluate_generation().
    """
    global _rouge
    if _rouge is None:
        nltk.download('punkt', quiet=True)
        _rouge = evaluate.load('rouge')

def evaluate_generation(reference, prediction):
    """
    Evaluates a predicted answer against the reference answer using BLEU-1 and ROUGE-L.
    Automatically initializes required resources on first use.

    Args:
        reference (str): Ground truth answer.
        prediction (str): Model-generated answer.

    Returns:
        dict: BLEU-1 score and ROUGE-L F1 score.
    """
    global _rouge
    if _rouge is None:
        _setup_metrics()

    # Tokenize for BLEU
    reference_tokens = nltk.word_tokenize(reference)
    prediction_tokens = nltk.word_tokenize(prediction)

    # BLEU-1: unigram weights
    smoothie = SmoothingFunction().method4
    bleu1_score = sentence_bleu(
        [reference_tokens],
        prediction_tokens,
        weights=(1, 0, 0, 0),  # BLEU-1
        smoothing_function=smoothie
    )

    # ROUGE-L
    rouge_result = _rouge.compute(predictions=[prediction], references=[reference])
    rouge_l_f1 = rouge_result['rougeL']

    return {
        'BLEU-1': bleu1_score,
        'ROUGE-L': rouge_l_f1
    }


# Example usage:
# reference_answer = "The mitochondria is the powerhouse of the cell."
# generated_answer = "Mitochondria are the powerhouses of cells."

# scores = evaluate_generation(reference_answer, generated_answer)
# print(scores)
