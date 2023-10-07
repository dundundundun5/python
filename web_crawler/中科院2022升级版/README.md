
# 爬虫说明
中科院期刊查询接口使用方法为：期刊名-id

id号需要手动查询

通过浏览器开发者工具能从中科院2022升级版的接口里看到期刊id号是

https://advanced.fenqubiao.com/Macro/PageData

https://advanced.fenqubiao.com/Macro/GetJson?start=1&length=20

https://advanced.fenqubiao.com/Macro/GetJson?start=21&length=20

但是这个接口无法直接访问，直接访问会被过滤器拦截，所以手动复制从接口返回的json数据，一共27页，每条上限20个，一共524个期刊，一共复制27次


http://advanced.fenqubiao.com/Journal/Detail/249639

上方链接是一个静态网页，罗列了一个期刊的所有信息，只需变换id，即可汇总不同期刊的具体信息


__包括计算机大类的分区，和最低档分区，所有分区信息罗列在"学科信息"列中__ 