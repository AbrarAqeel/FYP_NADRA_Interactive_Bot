import json
import datetime
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import torch

# --- Configuration ---
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
INPUT_FILE = r"C:\Users\PC\Desktop\AbrarAqeel\FYP\data\nadra_dataset.json"  # Matches your uploaded file name
OUTPUT_FILE = r"C:\Users\PC\Desktop\AbrarAqeel\FYP\data\nadra_dataset_translated.json"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


def translate_batch(tokenizer, model, texts):
    """
    Translates a batch of text strings from English to Urdu.
    """
    if not texts:
        return []

    # Use the modern tokenizer call instead of prepare_seq2seq_batch
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    # Generate translations
    with torch.no_grad():
        translated = model.generate(**batch)

    # Decode tokens back to text
    tgt_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_texts


def main():
    tokenizer, model = load_model()

    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_chunks = len(data)
    print(f"Found {total_chunks} chunks. Starting translation...")

    # Processing in batches with tqdm progress bar
    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Translating Batches", unit="batch"):
        batch = data[i: i + BATCH_SIZE]

        # 1. Extract text fields to translate
        # Use .get() safely; default to empty string if missing
        titles = [item.get("document_title", "") or "" for item in batch]
        headings = [item.get("section_heading", "") or "" for item in batch]
        texts = [item.get("text", "") or "" for item in batch]

        # Logic for summary: If summary is null, take first 200 chars of text to translate
        summaries_to_translate = []
        for item in batch:
            if item.get("summary"):
                summaries_to_translate.append(item["summary"])
            else:
                # Fallback: translate a short excerpt as the "Urdu Summary"
                summaries_to_translate.append(item.get("text", "")[:200])

        try:
            # 2. Perform Translations
            t_titles = translate_batch(tokenizer, model, titles)
            t_headings = translate_batch(tokenizer, model, headings)
            t_texts = translate_batch(tokenizer, model, texts)
            t_summaries = translate_batch(tokenizer, model, summaries_to_translate)

            # 3. Assign back to the JSON objects
            for idx, item in enumerate(batch):
                item["document_title_ur"] = t_titles[idx]
                item["section_heading_ur"] = t_headings[idx]
                item["text_ur"] = t_texts[idx]
                item["summary_ur"] = t_summaries[idx]

                # Update Metadata
                if "metadata" in item:
                    item["metadata"]["translation_status"] = "machine_translated"
                    item["metadata"]["translation_provider"] = MODEL_NAME
                    item["metadata"]["translation_date"] = datetime.datetime.now().isoformat()

        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            continue

    # Save output
    print(f"Saving translated data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
