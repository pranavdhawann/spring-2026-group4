import os

MAX_ITEMS = 10
OUTPUT_FILE = "dir_structure.txt"

def write_tree(path, file, prefix=""):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        file.write(prefix + "[Permission Denied]\n")
        return

    entries = entries[:MAX_ITEMS]

    for i, name in enumerate(entries):
        full_path = os.path.join(path, name)
        connector = "└── " if i == len(entries) - 1 else "├── "
        file.write(prefix + connector + name + "\n")

        if os.path.isdir(full_path):
            extension = "    " if i == len(entries) - 1 else "│   "
            write_tree(full_path, file, prefix + extension)

if __name__ == "__main__":
    start_dir = "."
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Directory structure of: {os.path.abspath(start_dir)}\n\n")
        write_tree(start_dir, f)

    print(f"Directory structure saved to {OUTPUT_FILE}")
