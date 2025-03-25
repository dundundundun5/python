# 已知视频的url，用户的cookie，爬取视频的cid和aid

def get_cid_n_aid(bv):
    import re, requests, os
    User_Agent = get_fake_user_agent()
    headers = {
        "User-Agent": User_Agent
    }
    url = f"https://www.bilibili.com/video/{bv}"
    r = requests.get(url=url, headers=headers)
    res = []
    # 在网页源代码中找视频的aid cid，经人工检验，第一个出现的就是所需答案
    for ch in ['a', 'c']:
        regex = re.compile(f'\"{ch}id\"[:][0-9]*')
        pos = re.search(regex, r.text)
        raw = r.text[pos.start():pos.end()]
        id = raw.split(":")[1]
        res.append(id)
    os.makedirs(f"D:/temp_files/{bv}", exist_ok=True)
    return res


def get_comments_n_save(bv):
    file = f'D:/temp_files/{bv}/{bv}_comments.csv'
    aid, cid = get_cid_n_aid(bv)
    import requests,re,time
    url = "https://comment.bilibili.com/" + cid + ".xml"
    User_Agent = get_fake_user_agent()
    headers = {
        "User-Agent": User_Agent
    }
    r = requests.get(url=url, headers=headers)

    lines = r.content.decode('utf-8')
    # 正则表达式筛选掉前面无关内容
    pattern = "\</source\>"
    temp = re.search(pattern=pattern, string=lines)
    pos = temp.end()
    # 找到无关内容的位置并取子串
    lines = lines[pos:]
    # 提取评论
    comment_message: list[str] = re.findall(pattern=">(.*?)</d>", string=lines, flags=re.I)
    # 提取弹幕参数
    params: list[list[str]] = re.findall(pattern="=\"\S+\">", string=lines, flags=re.I)
    # 用于存放转换前的数据
    raw_uids = []
    raw_times = []
    # 用于存放转换后的数据
    user_id_crc = []
    comment_datetime = []
    # 用逗号分割弹幕参数，并获取原始数据
    for param, i in params, len(params):
        print(f"拉取第{i}")
        a = re.split(',', string=param, maxsplit=0, flags=re.I)
        
        try:
            raw_uids.append(a[6])
        except IndexError:
            print(param)
        raw_times.append(a[4])
    print("进行数据处理，请等待。。。")
    # unix时间戳->本地时间
    for raw_time in raw_times:
        local_time = time.localtime(int(raw_time))
        res = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        comment_datetime.append(res)
    # uid和校验码的换算要另外进行
    user_id_crc = raw_uids
    # 换一种格式写入源文件
    import pandas as pd
    res =  pd.DataFrame(data=None, columns=['comment_datetime', 'user_id_crc', 'comment_message'])
    for i in range(len(comment_datetime)):
        res.loc[len(res)] = [comment_datetime[i],user_id_crc[i], comment_message[i]]
    res.to_csv(file, index=False)
    print(f"弹幕拉取完成，保存至{file}")


import binascii
# 计算uid（ascii）的crc循环校验码并转为十六进制

def crc32ascii(v):
    v = v.encode('utf-8')
    return '0x%8x' % (binascii.crc32(v) & 0xffffffff)

def get_fake_user_agent():
    from fake_useragent import UserAgent
    fake_ua = UserAgent()
    return fake_ua.chrome

def get_root_replies_n_save(bv):
    
    import requests
    oid, cid = get_cid_n_aid(bv)# oid就是aid
    file = f'D:/temp_files/{bv}/{bv}_replies_root.csv'
    root_file = f'D:/temp_files/{bv}/{bv}_root_id.txt'
    root = [int(oid)]
     
    i = 1 # 评论页码，一页含20层
    import pandas as pd
    res = pd.DataFrame(data=None,
                       columns=['parent_reply_id', 'user_id', 'user_id_crc',
                                'user_name', 'reply_datetime', 'reply_message', 'like'])
    while True:
        User_Agent = get_fake_user_agent()
        url = f'https://api.bilibili.com/x/v2/reply/main?jsonp=jsonp&next={i}&type=1&oid={oid}&mode=3&plat=1'
        headers = {
        'User-Agent':User_Agent
        }
        response = requests.get(url=url, headers=headers)
        raw_replies = response.json() # 获取dict格式的json响应
        if raw_replies['data']['cursor']['is_end']:
            break # 评论翻到底了
        raw_replies = raw_replies['data']['replies'] # 进入dict的评论数据区
        print(f"正在拉取第{i}页。。。", end='')
        for raw_reply in raw_replies: # 根据json分析出的规律，仅适用于b站视频评论
            root_reply_id = raw_reply['rpid']
            root.append(root_reply_id)
            user_id = raw_reply['member']['mid'] # 写作mid，读作uid
            user_id_crc = crc32ascii(user_id)
            user_id_crc = str(user_id_crc)[2:]
            user_name = raw_reply['member']['uname']
            import time
            r_datetime = raw_reply['ctime']
            local_time = time.localtime(int(r_datetime))
            reply_datetime = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
            reply_message = raw_reply['content']['message']
            like = raw_reply['like']
            # 装入pd.DataFrame
            data:list = [root_reply_id, user_id, user_id_crc, user_name, reply_datetime, reply_message, like]
            res.loc[len(res)] = data
        print("完成！")
        i = i + 1
    res.to_csv(file, index=False)
    with open(file=root_file, mode='w', encoding='utf-8') as f:
        # 列表转字符串，去掉前后方括号
        root = str(root)
        root = root[1:-1]
        # 写入文件，文件的形式是第一个数字是视频oid，后续是根评论的id
        f.write(root)

# 爬取子评论需要主评论的根ID，因此内部参数应为 oid, [root_id1, root_id2,...]
# 但是爬得太多了容易被反爬制裁
def get_sub_replies_n_save(bv):
    root_file = f'D:/temp_files/{bv}/{bv}_root_id.txt'
    file = f"D:/temp_files/{bv}/{bv}_replies_sub.csv"
    with open(root_file, mode='r', encoding='utf-8') as f:
        content = f.read()
        content = content.split(", ")
    oid = content[0]
    roots = content[1:]
    import requests
    oid, cid = get_cid_n_aid(bv)# oid就是aid
    i = 1 # 评论页码，一页含20层
    import pandas as pd
    columns = ['root_reply_id', 'user_id', 'user_id_crc',
                                'user_name', 'reply_datetime', 'reply_message', 'like']
    with open(file, mode='w', encoding="utf-8") as f:
        for i in range(len(columns)):
            if i == 0:
                f.write(columns[i])
            else:
                f.write(","+columns[i])
    res = pd.DataFrame(data=None,
                       columns=columns)
    for root in roots:       
        pn = 1
        # print(f"正在爬取评论{root}。。。", end="")
        while True:
            url = f'https://api.bilibili.com/x/v2/reply/reply?jsonp=jsonp&pn={pn}&type=1&oid={oid}&ps=10&root={root}&_=1647581648753'
            User_Agent = get_fake_user_agent()
            headers = {
            'User-Agent': User_Agent
            }
            response = requests.get(url=url, headers=headers)
            raw_replies = response.json() # 获取dict格式的json响应
            raw_replies = raw_replies['data'] # 进入dict的评论数据区
            if raw_replies == None:
                break;
            raw_replies =  raw_replies['replies']
            # try:
            #     for raw_reply in raw_replies:
            #         print(raw_reply)
            # except TypeError:
            #     print(type(raw_replies), raw_replies)
            #     exit(0)
            if raw_replies == None:
                break
            for raw_reply in raw_replies: # 根据json分析出的规律，仅适用于b站视频评论
                parent_reply_id = raw_reply['parent'] # 记录该评论依附的主评论
                user_id = raw_reply['member']['mid'] # 写作mid，读作uid
                user_id_crc = crc32ascii(user_id) # 计算32位循环校验后的uid
                user_id_crc = str(user_id_crc)[2:]
                user_name = raw_reply['member']['uname']
                import time
                r_datetime = raw_reply['ctime']
                local_time = time.localtime(int(r_datetime))
                reply_datetime = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
                reply_message = raw_reply['content']['message']
                like = raw_reply['like']
                data = [parent_reply_id, user_id, user_id_crc, user_name, reply_datetime, reply_message, like]
                res.loc[len(res)] = data
            res.to_csv(file, index=False, mode='a', header=False)
            print("chenggogn")  
            res = pd.DataFrame(data=None, columns=columns)
            pn += 1
        # print("完成！")
    print("全部完成！")

if __name__ == '__main__':
    bv = 'BV1kh4y1W7yW'
    #get_root_replies_n_save(bv)
    #get_sub_replies_n_save(bv)
    get_comments_n_save(bv)
    # import pandas as pd
    # file_c = f'D:/temp_files/{bv}_comments.csv'
    # file_r = f'D:/temp_files/{bv}_replies.csv'
    # comments = pd.read_csv(file_c)
    # replies = pd.read_csv(file_r)
    # input()
    