import pymysql
host = '116.204.72.94'
user = 'root'
password= 'WsAd12345!'
database = 'bilibili'
charset = 'utf8'
with pymysql.connect(host=host, user=user, password=password,
                     database=database, charset=charset) as db:
    cursor = db.cursor()
    sql = f"SELECT crc_uid from comments"
    try:
        cursor.execute(sql)
        db.commit()
        result = cursor.fetchall()
        print(result)
        input("")
    except:
        print("未写入成功")



