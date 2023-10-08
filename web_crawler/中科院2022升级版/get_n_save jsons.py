# "每天勤换Cookie"
import requests
SUBJECTS = {"地球科学":470,
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
YEAR = 2022
PATH = "./中科院2022升级版/json"
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.47",
    "Connection":"keep-alive",
    "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "Hm_lvt_bfa037370fbbd327cd70336871aea386=1696639957,1696743184; Hm_lvt_0dae59e1f85da1153b28fb5a2671647f=1696640098,1696743188; Hm_lpvt_0dae59e1f85da1153b28fb5a2671647f=1696743188; auth=53459D7F044D8737C78F3924D2610A2D1E73F8E12370AD0A298486EDB0CB5D6CB72E177EFA3E197675E3FE5604D22DC9C8C9396AB06A699B9366CCF4BEB9CF7D36FA69D7B599CE11436BFB8C22A9DAC87C829B4A27409DC67415E109FEAF105F66866E389B914DB08E2C50E66E8BD6320C29020AFDFF3BA6F73A5A29B710A6FD6FEA7F80E84277B67CF0CBA56363F404503E5ACB8F04598F7997D734F426BEC42862992AFE3D050DB8B3D0958FD82096F6B6DF4E976606D50C18EC7D1B4DD427B240A079A64E4726FE28FF27642F2AD0BFB8E8AD5CEF29A051EA16422AB56A2261A14F0D56ED6E97C42246A634AC414212A775221ACBF5A0C39ADC414FA2A8E56BFA12B7EB18D1F76DCB81D99EB60E270A9E7856B574D7B66B09FC259030BF41FF6B40C302F6B24D6456E8EEFE060FA5B43C209ADCC113F5245F297872F0924F15A7800034CC2289D7C3EF465B9B6204594E437F14A6F66294FA10FDBAAFC5EA9CB3478978A8B28DC9B4DD57F65F4B02E00E566D53567AE0C74941E143B0F7A8FC5D7013E98ABBA03CF33C60D7652792AA58FCAC46BDBF3B808ABB21AC48AFFC4CB4DEF3785A2F920F0036BB9353C3D4536B37395CA17E87FDFBD68A6ED14A6C; Hm_lpvt_bfa037370fbbd327cd70336871aea386=1696743664",
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
