import binascii
from tqdm import tqdm
import json
# 计算循环校验码
def crc32asii(v):
    return '%x' % (binascii.crc32(v) & 0xffffffff)
begin = 1
end = 9
res = []
# 打表计算循环校验码
for uid in tqdm(range(begin, end)):
    crc = crc32asii(bytes(str(uid), encoding='utf-8'))
    res.append({"crc":crc, "uid":uid})
# 转换成json字符串
json_res = json.dumps(res, ensure_ascii=False,indent=2)
# 写入本地
with open(file="tabele.json", mode="w", encoding="utf-8") as f:
    f.write(json_res)
    f.close()
