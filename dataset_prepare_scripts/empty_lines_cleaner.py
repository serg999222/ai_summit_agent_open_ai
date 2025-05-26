# clean

# with open("data_base.txt", "r", encoding="utf-8") as f:
with open("li.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Видалити порожні рядки та ті, що складаються лише з пробілів
cleaned_lines = [line for line in lines if line.strip() != ""]

# with open("data_base.txt", "w", encoding="utf-8") as f:
with open("li.txt", "r", encoding="utf-8") as f:
    f.writelines(cleaned_lines)
