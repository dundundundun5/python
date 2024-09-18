# "每天勤换Cookie"
import requests
SUBJECTS_2022 = {"地球科学":470,
            "物理与天体物理":323,
            "数学":543,
            "农林科学":702,
            "材料科学":410,
            "计算机科学":524,
            "环境科学与生态学":400,
            "化学":403,
            "工程技术":1183,
            "生物学":917,
            "医学":3722,
            "综合性期刊":63,
            "法学":858,
            "心理学":490,
            "教育学":260,
            "经济学":420,
            "管理学":423,
            "人文科学":444
            }
SUBJECTS_2023 = {"地球科学":470,
            "物理与天体物理":323,
            "数学":558,
            "农林科学":725,
            "材料科学":371,
            "计算机科学":501,
            "环境科学与生态学":388,
            "化学":389,
            "工程技术":1032,
            "生物学":940,
            "医学":3773,
            "综合性期刊":63,
            "社会学":1029,
            "心理学":488,
            "教育学":267,
            "经济学":413,
            "管理学":429,
            "哲学":416,
            "历史学":414,
            "文学":580,
            "艺术学":241
            }
SUBJECTS = SUBJECTS_2022
YEAR = 2022
PATH =f"./json_{YEAR}"


params = {
    "start":1,
    "length":-1
}
data = {
    "name":"None",
    "type":"zky",
    "keyword":"",
    "year":2022
}
url = "https://advanced.fenqubiao.com/Macro/GetJson"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.46",
    "Connection":"keep-alive",
    "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "Hm_lvt_bfa037370fbbd327cd70336871aea386=1696639957,1696743184,1697334760; Hm_lvt_7e6e274f8fab6425c743106cbd2e8d99=1709621244; Hm_lvt_0dae59e1f85da1153b28fb5a2671647f=1709621250; Hm_lpvt_0dae59e1f85da1153b28fb5a2671647f=1709621250; auth=80C32AD43D33C264498726A01FF85F7935A1FF4F8352FED423A573D4E32ADF0E15B08466B8E75787F5C81E95F6929EE98AB8A3A92F02F4EB35C1FCDC8B5664D4270F58FD6EE8FE784FAC3118B6820B2CB2967C0CA80A927D2F4CFE3E8DDE9C6DE30496CB9BCBC67B24B5347C3AD3B88A4ACC223B01772C8F75DB6B12ECCCB37AF5897539BF0FEFBA2A150A996AF522EAB93CC08E55863CCFA733F2A92541CD5EAB813638A9D74EBFEF83E0B5BF8D7A3B2894D12C5440A03253BA9EF38C1359B7BA803FFB0F2DF1AAA5060C484744585CC31168E06AF3CD3BA026D4EFC257FFE75A3BD442FAFCC32F38DE23FEAB2A6FE58D1CB218D5317F331BC290CA5CEB8A3968C5993BEF7DFB41AD4D76E7E924D5657C7BF8FDDCFE3BDB7E1C48EB5BA3BDA1D56769DFA0595F132486F7F4D7A47C2D697DD45406191BAD8447A520144DC4535441C6E51AE6F502C8B3E4412C8865EFB28DD515312F39C90BE3810CBEDED0F3E549007C29565492EB579531ACC839ABC7A2F025499A7D604202D6D03B7DFDD8538D03601B6FE3948F17094C315EAC9AF00FF89C84B3171D4077285C40FA1BDEFDA920158717EAF3726E47A1110D5855E295B3961AFDBF7C570FF50E0579D62712D851F03143CF6D9E948B930C013880; Hm_lpvt_7e6e274f8fab6425c743106cbd2e8d99=1709621375",
    "Host":"advanced.fenqubiao.com",
    }
import os
os.makedirs(PATH, exist_ok=True)
for key,value in SUBJECTS.items():
    params['length'] = value # 要爬多少条期刊
    data["name"] = key
    data["year"] = YEAR
    r = requests.post(url=url,headers=headers, params=params, data=data)
    with open(f"{PATH}/{key}.json", mode='w',encoding='utf-8') as f:
        f.write(r.text)
