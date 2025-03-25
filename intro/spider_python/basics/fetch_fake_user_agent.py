from fake_useragent import UserAgent
# 实例化一个对象
user_agent = UserAgent()
# 随机获取一个
print(user_agent.chrome)
print(user_agent.firefox)
