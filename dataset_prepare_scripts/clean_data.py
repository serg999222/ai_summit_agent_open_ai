import os
import re

# def clean_text(content):
#     lines = content.splitlines()
#     cleaned = []
#     buffer = ""

#     for line in lines:
#         line = line.strip()

#         # Пропуск номерів та таймкодів
#         if re.match(r"^\d+$", line):
#             continue
#         if re.match(r"^\d\d:\d\d:\d\d\.\d+ -->", line):
#             continue
#         if line == "":
#             if buffer:
#                 cleaned.append(buffer.strip())
#                 buffer = ""
#             continue

#         buffer += " " + line

#     if buffer:
#         cleaned.append(buffer.strip())

#     return "\n\n".join(cleaned)

def clean_text(content):
    lines = content.splitlines()
    cleaned = []
    date_line = ""
    output_buffer = []

    # 1. Знаходимо перший непорожній рядок (дату)
    for line in lines:
        if line.strip():
            date_line = line.strip()
            break

    current_timecode = None

    for line in lines:
        line = line.strip()

        # Пропускаємо номери рядків
        if re.match(r"^\d+$", line):
            continue

        # Таймкод
        if re.match(r"^\d\d:\d\d:\d\d\.\d+ -->", line):
            current_timecode = line
            output_buffer.append(f"{date_line}   {current_timecode}")
            continue

        # Порожній рядок завершує поточний блок
        if line == "":
            if output_buffer and output_buffer[-1] != "":
                output_buffer.append("")
            continue

        # Додаємо текст, якщо це не таймкод і не номер
        if current_timecode:
            output_buffer.append(line)
        else:
            # Ігноруємо текст до першого таймкоду
            continue

    # Прибираємо зайві пусті рядки
    # cleaned = [s for s in output_buffer if s.strip() != ""]
    cleaned = output_buffer
    return "\n".join(cleaned)


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

