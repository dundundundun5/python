import json
PATH = "./中科院2022升级版/"
SUBJECT = "计算机科学"
def read_json(path):
    f = open(path + '手动.json', 'r')
    content = f.read()
    a = json.loads(content)
    result_json = []
    for i in range(len(a)):
        
        for j in range(len(a[i])):
            result_json.append(a[i][j]) # a[i][j]['id']是我们要的id-期刊名
    # print(len(result_json))
    return result_json

def get_pages_by_id(id):
    url = f"https://advanced.fenqubiao.com/Journal/Detail/{id}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.47",
    "Connection":"keep-alive",
    "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "Hm_lvt_bfa037370fbbd327cd70336871aea386=1696639957; Hm_lvt_0dae59e1f85da1153b28fb5a2671647f=1696640098; Hm_lpvt_0dae59e1f85da1153b28fb5a2671647f=1696640175; auth=10748E21932E558785CF16F5088024CC2ACC2F09541DA9912B659D6915700704D37D90E79311E5564E1AB890B25A3D11B7E3E6FCCE2C67DA8C0241CB2080D2D94A719FB79EA0FEB957A767CE32080544458221D248C65F31CFAA552AA439D164690C71840411DF5C54AF9FD945B7059C5C341058C22927D0EF4A34D94DB6529F0BEB08A7B47105ECC801BF082BA55D8A9D3CF86AA3400E8E45374D78493CEE764D1668BF53EB1998B466C2C7722AB295F51ABC5C7598970A3231E57C2DB2022299BFBE40AFD11CA303D1163D2158AD18D929702F3784A9D6EB4EC657AB8B36D81F0C222A241863CF94CAB435AFB95A6C817069ED391A01229C9D5C627C42A656244B42E9B51BCEBABC90A6177557FAFACFBEA9B6811496D2C22D10850436914439262B33F625189BB23EEBA6B9C61ACCA49F67AE5A406CD3C0AB6E0AEE529EDD477FB57C879DD063CE4C91A15FBD5A8C3333661F260CEE39E9CBFB2A691D22E7E1382976E25008FB022417131A99EECE29D2E83F3B954E35859135F00224BC3E5333D34BC5548D13C7BE1083037115947316A217C5F8FB102BF71B87428DEECC00547664CDFCFD906B74CE4566747D2810E35F103833E0A592D6EAB294D9758E; Hm_lpvt_bfa037370fbbd327cd70336871aea386=1696644227",
    "Host":"advanced.fenqubiao.com",
    "Referer":"https://advanced.fenqubiao.com/Macro/Journal?name=%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%91%E5%AD%A6&year=2022",
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
    # print(info)
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
    res_csv = []
    result_json = read_json(PATH)
    # print(result_json)
    
    for i in range(len(result_json)):
        zky_journal_id = result_json[i]['id']
        t = get_pages_by_id(zky_journal_id)
        info = get_info(t)
        info["id"] = zky_journal_id
        # print(info)
        # 制作视图，重新整理数据
        csv = [info["刊名"],info["id"],int(info["年份"]),info["分区"],info["ISSN"],info["Review"],info["Open Access"],info["最低档"],info["学科信息"]]
        res_csv.append(csv)
        print(csv)
    res_csv = pd.DataFrame(res_csv, columns=columns)
    print(res_csv.head())
    res_csv.to_excel(PATH + "中科院2022升级版_爬虫初版.xlsx", index=False)
