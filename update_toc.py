import re


def main():
    with open("README.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    toc = ["* Template\n", "* Flags\n", "* Pragmas\n", "* Debug\n"]
    sections = {"Algoritmos", "Estruturas", "Utils", "Fatos"}
    capturing = False

    for line in lines:
        if m := re.match(r"^(#{1,3})\s+(.+)$", line.strip()):
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                capturing = title in sections
                if capturing:
                    toc.append(f"* {title}\n")
            elif capturing:
                toc.append(f"{'  ' * (level - 1)}* {title}\n")

    out = []
    skipping = False
    for line in lines:
        s = line.strip()
        if s == "# Sumário":
            out.append(line)
            out.append("\n")
            out.extend(toc)
            out.append("\n")
            skipping = True
        elif s == "### Template":
            skipping = False

        if not skipping:
            out.append(line)

    with open("README.md", "w", encoding="utf-8") as f:
        f.writelines(out)


if __name__ == "__main__":
    main()
