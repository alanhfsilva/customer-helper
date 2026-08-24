# Evaluation Rubric

## Accuracy (0.0–1.0)

- **1.0**: Answer is factually correct and covers all key points from the reference.
- **0.8**: Answer is mostly correct with minor omissions.
- **0.5**: Answer is partially correct but missing significant information.
- **0.2**: Answer contains some correct elements but mostly wrong or incomplete.
- **0.0**: Answer is completely wrong or irrelevant.

## Faithfulness (0.0–1.0)

- **1.0**: Every claim is directly supported by the provided context.
- **0.8**: Most claims supported; minor embellishments that don't mislead.
- **0.5**: Some claims supported, others not clearly grounded.
- **0.0**: Claims are fabricated or contradict the context.

## Citation Validity (0.0–1.0)

- **1.0**: All cited sources exist and support the claims they annotate.
- **0.5**: Some citations valid, others reference wrong or non-existent sources.
- **0.0**: No valid citations or no citations provided when claims are made.

## Refusal Correctness

- **Correct refusal**: Assistant abstains on out-of-scope questions.
- **Incorrect refusal**: Assistant abstains on in-scope questions it should answer.
- **Missing refusal**: Assistant answers out-of-scope questions it should refuse.
