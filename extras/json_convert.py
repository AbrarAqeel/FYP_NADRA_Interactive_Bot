import json
import datetime
import re

# Configuration
INPUT_FILE = r'data\NADRA_Cleaned_Dataset.txt'
OUTPUT_FILE = r'data\nadra_dataset.json'

# Metadata Constants (matches your desired format)
METADATA_TEMPLATE = {
    "source_file": "NADRA_Cleaned_Dataset.txt",
    "source_citation": "NADRA_Cleaned_Dataset.txt",
    "date_collected": datetime.date.today().isoformat(),
    "version": "v1.0",
    "language": "en+ur",
    "country": "Pakistan",
    "translation_status": "pending",
    "translation_provider": None,
    "needs_human_review": False,
    "translation_date": None
}


def calculate_tokens(text):
    """Simple whitespace token count approximation."""
    if not text:
        return 0
    return len(text.split())


def create_chunk_object(chunk_id, doc_title, section_heading, text, category="Identity Services"):
    """Creates a single JSON object based on the schema."""

    # Create a unique ID (e.g., chunk_001_00001)
    formatted_id = f"chunk_{chunk_id:03d}"

    current_time = datetime.datetime.now().isoformat()

    return {
        "id": formatted_id,
        "document_title": doc_title,
        "category": category,
        "subcategory": None,
        "section_heading": section_heading,
        "text": text.strip(),
        "text_ur": None,
        "summary": None,
        "tokens": calculate_tokens(text),
        "metadata": METADATA_TEMPLATE,
        "created_at": current_time,
        "document_title_ur": None,
        "section_heading_ur": None,
        "summary_ur": None
    }


def parse_nadra_file(file_path):
    chunks = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # State variables
    current_doc_title = "General"
    current_section_heading = "Introduction"
    current_text_buffer = []

    chunk_counter = 1

    # Regex patterns for separators
    # Matches lines that are mostly === or ---
    re_title_sep = re.compile(r'^={10,}$')
    re_section_sep = re.compile(r'^-{10,}$')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines if we haven't started collecting text
        if not line and not current_text_buffer:
            i += 1
            continue

        # 1. CHECK FOR DOCUMENT TITLE (surrounded by ======)
        if re_title_sep.match(line):
            # If we have accumulated text, save previous chunk first
            if current_text_buffer:
                full_text = " ".join(current_text_buffer)
                if full_text.strip():
                    chunks.append(
                        create_chunk_object(chunk_counter, current_doc_title, current_section_heading, full_text))
                    chunk_counter += 1
                current_text_buffer = []

            # The next line should be the title
            if i + 1 < len(lines):
                current_doc_title = lines[i + 1].strip()
                current_section_heading = "General"  # Reset section on new document
                i += 2  # Skip the separator and the title line

                # Check if there is a closing separator (optional but likely)
                if i < len(lines) and re_title_sep.match(lines[i].strip()):
                    i += 1
                continue

        # 2. CHECK FOR SECTION HEADING (surrounded by ------)
        elif re_section_sep.match(line):
            # If we have accumulated text, save previous chunk first
            if current_text_buffer:
                full_text = " ".join(current_text_buffer)
                if full_text.strip():
                    chunks.append(
                        create_chunk_object(chunk_counter, current_doc_title, current_section_heading, full_text))
                    chunk_counter += 1
                current_text_buffer = []

            # The next line should be the section heading
            if i + 1 < len(lines):
                current_section_heading = lines[i + 1].strip()
                i += 2  # Skip separator and heading

                # Check for closing separator
                if i < len(lines) and re_section_sep.match(lines[i].strip()):
                    i += 1
                continue

        # 3. NORMAL TEXT
        else:
            # Add line to buffer
            if line:
                current_text_buffer.append(line)
            i += 1

    # Save the final chunk if exists
    if current_text_buffer:
        full_text = " ".join(current_text_buffer)
        if full_text.strip():
            chunks.append(create_chunk_object(chunk_counter, current_doc_title, current_section_heading, full_text))

    return chunks


# --- Main Execution ---
try:
    print(f"Reading from {INPUT_FILE}...")
    json_data = parse_nadra_file(INPUT_FILE)

    print(f"Generated {len(json_data)} chunks.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully wrote JSON to {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Error: Could not find file {INPUT_FILE}. Make sure the path is correct.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
