import pymysql
host = 'localhost'
user = 'root'
password= 'wsad12345'
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



