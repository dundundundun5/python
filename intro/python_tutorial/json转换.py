import json
# json是一种轻量级的数据交互格式，本质是字符串
data = [{"name": "xqc", "age": 14}, {"name": "pvc", "age": 24},
        {"name": "lhc", "age": 34}]
# 不用ascii码 字典列表转json
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)
# json字符串转字典列表
s= '[{"name": "xqc", "age": 14}, {"name": "pvc", "age": 24}, {"name": "lhc", "age": 34}]'
l = json.loads(s)
print(l)