# "勤换cookie"
import json
FINAL_PATH = "./results/"
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
SUBJECTS = SUBJECTS_2023
YEAR = 2023

SUBJECT = ""
urlencode_subject = ""
PATH = f"./json_{YEAR}"
import os
os.makedirs(FINAL_PATH, exist_ok=True)
def read_json(path, subject):
    f = open(f'{path}/{subject}.json', 'r')
    content = f.read()
    a = json.loads(content)
    result_json = []
    for i in range(len(a)):
        result_json.append(a[i]['Id'])
    # print(len(result_json))
    return result_json

def get_pages_by_id(id):
    url = f"https://advanced.fenqubiao.com/Journal/Detail/{id}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.46",
    "Connection":"keep-alive",
    "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "Hm_lvt_bfa037370fbbd327cd70336871aea386=1696639957,1696743184,1697334760; Hm_lvt_7e6e274f8fab6425c743106cbd2e8d99=1709621244; Hm_lvt_0dae59e1f85da1153b28fb5a2671647f=1709621250; Hm_lpvt_0dae59e1f85da1153b28fb5a2671647f=1709621250; auth=80C32AD43D33C264498726A01FF85F7935A1FF4F8352FED423A573D4E32ADF0E15B08466B8E75787F5C81E95F6929EE98AB8A3A92F02F4EB35C1FCDC8B5664D4270F58FD6EE8FE784FAC3118B6820B2CB2967C0CA80A927D2F4CFE3E8DDE9C6DE30496CB9BCBC67B24B5347C3AD3B88A4ACC223B01772C8F75DB6B12ECCCB37AF5897539BF0FEFBA2A150A996AF522EAB93CC08E55863CCFA733F2A92541CD5EAB813638A9D74EBFEF83E0B5BF8D7A3B2894D12C5440A03253BA9EF38C1359B7BA803FFB0F2DF1AAA5060C484744585CC31168E06AF3CD3BA026D4EFC257FFE75A3BD442FAFCC32F38DE23FEAB2A6FE58D1CB218D5317F331BC290CA5CEB8A3968C5993BEF7DFB41AD4D76E7E924D5657C7BF8FDDCFE3BDB7E1C48EB5BA3BDA1D56769DFA0595F132486F7F4D7A47C2D697DD45406191BAD8447A520144DC4535441C6E51AE6F502C8B3E4412C8865EFB28DD515312F39C90BE3810CBEDED0F3E549007C29565492EB579531ACC839ABC7A2F025499A7D604202D6D03B7DFDD8538D03601B6FE3948F17094C315EAC9AF00FF89C84B3171D4077285C40FA1BDEFDA920158717EAF3726E47A1110D5855E295B3961AFDBF7C570FF50E0579D62712D851F03143CF6D9E948B930C013880; Hm_lpvt_7e6e274f8fab6425c743106cbd2e8d99=1709621375",
    "Host":"advanced.fenqubiao.com",
    "Referer":f"https://advanced.fenqubiao.com/Macro/Journal?name={urlencode_subject}&year={YEAR}",
    }
    import requests
    r = requests.get(url=url, headers=headers)
    # print(r.text)
    return r.text

def get_style_classes(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "lxml")
    
    styles = soup.find("style")
    import re
    re1 = re.findall("[a-z][0-9]", styles.get_text())
    re2 = re.findall("\'[0-9]\'", styles.get_text())
    #print(re1)
    for i in range(len(re2)):
        re2[i] = re2[i][1]
    #print(re2)
    res = {}
    for i in range(len(re2)):
        res[re1[i]] = re2[i]
    return res

def get_info(t):
    style_classes = get_style_classes(t)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(t, "lxml")
    tb1 = soup.findAll("table")[0] 
    tb2 = soup.findAll("table")[1] # 包含大类小类的分区，比较难归类
    info = {}
    for i in range(5):
        tr = tb1.find_all("tr")[i]
        name = tr.find_all("td")[0].get_text()
        value = tr.find_all("td")[1].get_text()  
        info[name] = value  
    tb2_tr = tb2.find_all("tr")
    res = {}
    lowest = 0
    for i in range(len(tb2_tr)):
        if i == 0:
            continue
        tb2_tr_td = tb2_tr[i].findAll("td")
        section,rank = None,None
        for j in range(len(tb2_tr_td)):
            
            if tb2_tr_td[j].a !=  None:
                # print(tb2_tr_td[j].a.get_text())
                section = tb2_tr_td[j].a.get_text()
            if tb2_tr_td[j].span != None:
                # print(tb2_tr_td[j].span['class'][0])
                rank = tb2_tr_td[j].span['class'][0]
                if lowest < int(style_classes[rank]):
                    lowest = int(style_classes[rank])
        res[section] = style_classes[rank]
        # print("==========")
    info["分区"] = int(res[SUBJECT])
    info["最低档"] = lowest
    info["学科信息"] = res
    return info
if __name__ == '__main__':

    import pandas as pd
    columns=["刊名","中科院ID","年份","分区","ISSN","综述","开放","最低档","学科信息"]
    
    for key,value in SUBJECTS.items():
        res_csv = []
        SUBJECT = key
        result_json = read_json(PATH, SUBJECT)
        for i in range(len(result_json)):
            zky_journal_id = result_json[i] - 9
            t = get_pages_by_id(zky_journal_id)
            info = get_info(t)
            info["id"] = zky_journal_id
            # 制作视图，重新整理数据
            csv = [info["刊名"],info["id"],int(info["年份"]),info["分区"],info["ISSN"],info["Review"],info["Open Access"],info["最低档"],info["学科信息"]]
            res_csv.append(csv)
            print(csv)
        res_csv = pd.DataFrame(res_csv, columns=columns)
        print(res_csv.head())
        res_csv.to_excel(f"{FINAL_PATH}/中科院{YEAR}升级版_{SUBJECT}.xlsx", index=False)
        
