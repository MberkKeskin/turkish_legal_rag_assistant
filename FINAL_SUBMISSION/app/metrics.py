
import re
import string
from collections import Counter


def normalize_answer_tr(text):
    text = str(text).lower()
    text = text.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
    text = text.replace("ş", "s").replace("ö", "o").replace("ç", "c")
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def exact_match(pred, gold):
    return int(normalize_answer_tr(pred) == normalize_answer_tr(gold))


def token_f1(pred, gold):
    pred_tokens = normalize_answer_tr(pred).split()
    gold_tokens = normalize_answer_tr(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def simple_faithfulness(pred, context):
    pred_tokens = set(normalize_answer_tr(pred).split())
    context_tokens = set(normalize_answer_tr(context).split())

    if not pred_tokens:
        return 0.0

    return len(pred_tokens & context_tokens) / len(pred_tokens)


def clean_gold_answer_for_eval(gold):
    gold = str(gold)
    gold = gold.replace("Kaynağa göre:", "").strip()
    gold = re.split(r"\n\s*Kaynak\s*:", gold)[0].strip()
    gold = re.split(r"Kaynak\s*:", gold)[0].strip()
    return gold.strip()


def rubric_score_from_answer_df(df):
    R = df["retrieval_hit"].mean()
    A = (df["F1"] * df["retrieval_hit"]).mean()
    G = (df["Faithfulness"] * df["retrieval_hit"]).mean()

    final = 0.35 * R + 0.40 * A + 0.25 * G

    result = {
        "Retrieval_R": R,
        "Answer_A_penalized_F1": A,
        "Grounding_G_penalized_Faithfulness": G,
        "Final_Rubric_Score": final,
        "Raw_F1": df["F1"].mean(),
        "Raw_Faithfulness": df["Faithfulness"].mean(),
        "EM": df["EM"].mean(),
    }

    if "Citation_Accuracy" in df.columns:
        result["Citation_Accuracy"] = df["Citation_Accuracy"].mean()

    return result
