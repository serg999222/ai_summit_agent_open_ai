import os
import re

def clean_text(content):
    lines = content.splitlines()
    cleaned = []
    buffer = ""

    for line in lines:
        line = line.strip()

        # Пропуск номерів та таймкодів
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"^\d\d:\d\d:\d\d\.\d+ -->", line):
            continue
        if line == "":
            if buffer:
                cleaned.append(buffer.strip())
                buffer = ""
            continue

        buffer += " " + line

    if buffer:
        cleaned.append(buffer.strip())

    return "\n\n".join(cleaned)

def process_transcripts(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(input_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    cleaned = clean_text(content)
                    outfile.write(cleaned + "\n\n")

# 🚀 Виклик функції:
process_transcripts("raw_transcript", "data_base.txt")

