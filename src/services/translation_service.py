from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define language codes and model name for translation
src_lang, tgt_lang = "eng_Latn", "ben_Beng"
trans_model_name = "ai4bharat/indictrans2-en-indic-1B"

# Load tokenizer and model for IndicTrans
tokenizer_trans = AutoTokenizer.from_pretrained(trans_model_name, trust_remote_code=True)
model_trans = AutoModelForSeq2SeqLM.from_pretrained(
    trans_model_name, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE)

# Initialize the IndicProcessor
ip = IndicProcessor(inference=True)

def translate_sentences(input_sentences):
    """Translates input sentences from English to Bengali."""
    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer_trans(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_tokens = model_trans.generate(
            **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5
        )

    translations = tokenizer_trans.batch_decode(generated_tokens.cpu().tolist(), skip_special_tokens=True)
    return ip.postprocess_batch(translations, lang=tgt_lang)
