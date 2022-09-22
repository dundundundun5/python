#
# with open(file="F:\\python\\intro\\test.txt",
#           mode="r", encoding="UTF-8") as f:
#     print(f.read().count("ma"))
src = open(file="F:\\python\\intro\\test.txt",
        mode="r", encoding="UTF-8")
lines = src.read()
dest = open(file="F:\\python\\intro\\备份.txt", mode="w", encoding="UTF-8")
dest.write(lines)
dest.flush()
src.close()
dest.close()