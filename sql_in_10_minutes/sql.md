# SQL必知必会
重点学习MYSQL，SQL不区分大小写，注意保持代码可读性
## 8 使用函数处理数据
1. 文本函数

|函数|用途|
|-|-|
|SUBSTRING()|提取字符串的组成部分|
|LEFT()/RIGHT()|返回字符串左边/右边的字符|
|LENGTH()|返回字符串的长度|
|LOWER()/UPPER()|转换为小写/大写|
|LTRIM()/RTRIM()|去除字符串左边/右边的空格|
|SOUNDEX()|返回字符的SOUNDEX值,用于发音相似的字符串|

```
SELECT vend_name, UPPER(vend_name) AS vend_name_upcase
FROM Vendors
ORDER BY vend_name;
```
    选择两列vend_name和大写的vend_name，修改后者的命名。
    最后按vend_name排序
```
SELECT cust_name, cust_contact FROM Customers
WHERE SOUNDEX(cust_contact) = SOUNDEX('Michael Green');
```
    从Customers选择两列，但条件是cust_contact的soundex值是Michael Green
    （结果返回Michelle Green）
2. 日期函数

|函数|用途|
|-|-|
|YEAR()|提取日期的年|
|MONTH()|提取日期的月|
|MONTH()|提取日期的日|
```
SELECT order_num
FROM Orders
WHERE YEAR(order_date) = 2020;
```
    从X选择XX，条件是，order_date的年份为2020
3. 数值处理函数

|函数|
|-|
|ABS()|
|COS()|
|EXP()|
|PI()|
|SIN()|
|SQRT()|
|TAN()|

    函数自解释，仅记录

## 9 汇总数据

1. 汇总非检索

    1. 确定表中的行数（带条件）
    2. 获得某些行的和（带条件）
    3. 找出表列行的最大、最小、平均值（带条件）
2. 用于汇总的聚集函数

|函数|用途|
|-|-|
|AVG()|某列的平均值（忽略NAN）|
|COUNT()|某列的行数|
|MAX()|某列的最大值（忽略NAN）|
|MIN()|某列的最小值（忽略NAN）|
|SUM()|某列值之和（忽略NAN）|

```
SELECT AVG(prod_price) AS avg_price
FROM Products
WHERE vend_id = 'DLL01';
```
    选择某列的平均值（带条件）
```
SELECT COUNT(*) AS num_cust
FROM Customers;
```
    计算表中的总行数
```
SELECT COUNT(cust_email) as num_cust
FROM Customers;
```
    计算某列的有效行数，排除列中的NAN行
```
SELECT MAX(prod_price) as max_price
FROM Products;
SELECT MIN(prod_price) as max_price
FROM Products;
```
    计算某列数值的最大值
    计算某列数值的最小值
```
SELECT SUM(quantity) AS items_ordered
FROM OrderItems
WHERE order_num = 20005;
```
    条件选取某一列，并计算列的总和值
2. 聚集不同值

|参数|用途|
|-|-|
|ALL|对所有行执行计算（默认）|
|DISTINCT|只包含不同的值（需指定）|

* DISTINCT只能用于列名，而不是计算或表达式

```
SELECT AVG(DISTINCT prod_price) AS avg_price
FROM Products
WHERE vend_id = 'DLL01';
```
    DISTINCT只选择列中不同的数值
```
SELECT COUNT(*) AS num_items,
MIN(prod_price) AS price_min,
MAX(prod_price) AS price_max,
AVG(prod_price) AS price_avg
FROM Products;
```
    执行四条聚集运算，同时返回四个值

## 10 分组数据


目前的条件仍然基于WHERE

分组的关键词提供了更加高级的条件搜索

将数据分为多个逻辑组，对每个组进行聚集计算

1. GROUP BY
```
SELECT vend_id, COUNT(*) AS num_prods
FROM Products
GROUP BY vend_id;
```
    对表按vend_id进行分组，
    然后对每个组提取vend_id以及聚集运算出每个组的总行数
* GROUP BY可以包含任意多列，因此可以嵌套
* 嵌套分组后，会在所有分组上进行运算，而不能从个别列取回数据
* GROUP BY不能用于别名
* 除了聚集计算外，SELECT语句中的每一列都必须在GROUP BY子句中给出
* WHERE GROUP BY ORDER BY
* 列中的NULL分为同组
* GROUP BY 2,1表示可按选择的第二个列分组，再按第一列分组

2. HAVING

WHERE只能过滤指定行，而不能过滤分组

WHERE在分组前过滤，HAVING在数据分组后过滤

<font color='yellow'>
WHERE管的是所有行分组前的通病

HAVING管的是按列分组后，分组们的通病
</font>

```
SELECT vend_id, COUNT(*) AS num_prods
FROM Products
WHERE prod_price >= 4
GROUP BY vend_id
HAVING COUNT(*) >= 2;
```
    where先过滤掉所有行中prod_price小于4的行，然后对vend_id进行分组，
    筛除总行数小于2的分组
3. ORDER BY & GROUP BY

无论分组还是未分组，ORDER BY总是管住最终输出的顺序问题

```
SELECT order_num, COUNT(*) AS items
FROM OrderItems
GROUP BY order_num
HAVING COUNT(*) >= 3 
ORDER BY items, order_num
```
    除了聚集计算外，SELECT语句中的每一列都必须在GROUP BY子句中给出
    HAVING也可以用items别名
    从表中选出两列，order_num和总行数，然后对order_num进行分组
    并去除分组中总行数小于3的分组，最后对获胜分组们先按items后按
    order_num排序，输出最终结果
4. 顺序总结

|子句|说明|必要性|
|-|-|-|
|SELECT|要返回的列或表达式|是|
|FROM|从中检索数据的表|仅在从表选择数据时使用|
|WHERE|行级过滤|否|
|GROUP BY|分组说明|仅在计算聚集时使用|
|HAVING|组级过滤|否|
|ORDER BY|输出排序|否|


## 11 使用子查询

目前见到的都是单表查询，然而

如果需要找订购了某物品的所有顾客，就无法只用单表查询完成任务，规则如下：

1. Orders存储了所有订单，订单包含顾客ID，订单ID，日期等
2. OrdersItems存储了每条订单的物品集合
3. Customers存储了顾客ID对应的详细信息
---
那么查询也分为嵌套的三步:

1. 从OrderItems 找到所有包含该物品的订单ID
2. 拿着检索出的订单ID，从Orders找到所有这些订单ID对应的顾客ID
3. 拿着检索出的顾客ID，从Customers里找到所有这些顾客ID对应的详细信息
---
1. 嵌套查询


反过来，可以看作递归
$$详细信息集\rightarrow顾客ID集\rightarrow订单ID集$$

```
SELECT cust_name, cust_contact
FROM Customers
WHERE cust_id IN (
		SELECT cust_id 
		FROM Orders
		where order_num IN (
				SELECT order_num
				FROM OrderItems
				WHERE prod_id = 'RGAN01'
		)
);
```
    用where子句嵌套子查询
    相比于以往的单表查询，此次查询耗费19s
2. 计算字段

如果要显示Customers表中每个顾客的订单总数，而订单与相应的顾客ID存储在Orders表

突破口是，两个表中共有的顾客ID，沿着顾客ID取Orders表中查询
```
SELECT
	cust_name,
	(SELECT COUNT(*)
	 FROM Orders
	 WHERE Orders.cust_id = Customers.cust_id) as orders
FROM Customers
ORDER BY cust_name
```
    orders是在Orders表中统计行数，条件是仅当表中顾客id在Customers表中存在时才统计

## 12/13 联结表
由于没有学过关系表和关系数据库

需要先补充 关系表和关系数据库

关系表<https://blog.csdn.net/YXXXYX/article/details/123270424>

ER图<https://blog.csdn.net/caohongxing/article/details/122398825>

在本例中，供应商信息表和产品信息表是多对一的关系，多个产品可以对应到一个供应商，一个产品只有一个供应商

那么则需要一个外键来联系两张表，比如供应商ID
1. 内联结

    内联结又叫做等值联结

```
SELECT vend_name, prod_name, prod_price
FROM Vendors
INNER JOIN Products ON Vendors.vend_id = Products.vend_id
```
    从FROM开始，表示Vendors内联结Products,ON后的语句表示两表联结的条件
```
SELECT cust_name, cust_contact
FROM Customers AS C, Orders AS O, OrderItems AS OI
WHERE C.cust_id = O.cust_id 
AND		OI.order_num = O.order_num
AND 	prod_id = 'RGAN01'
```
    AS是表的别名
    首先用外键联结Customers和Orders，同样用外键联结Orders和OrderItems，
    最后在OrderItemss中提供过滤条件

**联结多对一的父子表时，两张表的外键名不一定是相同的（.cust_id）** 
2. 自联结

使用表别名可以在一条SELECT中引用多次同一张表

一张表与自己的联结，称为内联结

这种情况下，表与自己相当于是一对一情况中的共享主键

```
SELECT C1.cust_id, C1.cust_name, C1.cust_contact
FROM Customers AS C1, Customers AS C2
WHERE C1.cust_name = C2.cust_name
AND	C2.cust_contact = 'Jim Jones'
```

    首先用公司名作为联结主键，先添加筛选找到Jim Jones的公司，从而找到该公司下的所有人
3. 自然联结

标准的联结返回所有数据，相同的列会多次出现

自然联结排除多次出现，要求只能选择那些唯一的列（手动选择）

```
SELECT C.*, O.order_num, O.order_date, O.cust_id,
			OI.prod_id, OI.quantity, OI.item_price
FROM Customers AS C, Orders AS O, OrderItems AS OI
WHERE C.cust_id = O.cust_id 
AND		OI.order_num = O.order_num
AND 	prod_id = 'RGAN01'
```
    事实上，几乎所有内联结都是自然联结，很可能永远都不会用到非自然联结的内联结
4. 外联结

用于查看没有关联的行，比如未下订单的顾客，无人订购的产品

```
SELECT C.cust_id, O.order_num
FROM Customers as C
LEFT OUTER JOIN Orders AS O ON C.cust_id = O.cust_id
```
    检索包括了没有订单的客户
    OUTER JOIN表示外联结 LEFT和RIGHT表示左外联结和右外联结
    唯一差别是所关联表的顺序
    如果是左联结，则保留左表所有的行；即使在右表中没有匹配的行
    左右联结可以互相转换
5. 聚集+联结

    联结后的表可以用聚集函数进行统计
```
SELECT Customers.cust_id,COUNT(Orders.order_num) as num_prod
FROM Customers
INNER JOIN Orders ON Customers.cust_id = Orders.cust_id
GROUP BY Customers.cust_id
```
    首先用顾客ID联结两张表，随后按顾客ID分组，对每个顾客的订单计数
```
SELECT Customers.cust_id,COUNT(Orders.order_num) as num_prod
FROM Customers
LEFT OUTER JOIN Orders ON Customers.cust_id = Orders.cust_id
GROUP BY Customers.cust_id
```
    外联结同理
## 14 组合查询
用于把多条SELECT语句合在一起形成同一条语句，
1. UNION

    规则简述：

        1. 根据DBMS文档，查看UNION能组合的最大语句数目
        2. 一个必须用于分割两段Select语句
        3. 每个查询必须包含相同的列、表达式、聚集函数，但顺序可以不同
        4. UNION ALL可以指定显示重复查询的（默认过滤相同的行）
        5. 如果两个查询的列名不一样，则只有第一个会生效
        6. ORDER BY只能在最后用一次，所以不存在两个SELECT各自排序的情况
```
SELECT cust_name, cust_contact, cust_email
FROM Customers
WHERE cust_state IN ('IL','IN','MI')
UNION
SELECT cust_name, cust_contact, cust_email
FROM Customers
WHERE cust_name = 'fun4All'
ORDER BY cust_name, cust_contact
```
    由UNION连接的两组SELECT语句

# 15 插入数据
数据插入方式
* 插入完整的行
* 插入行的一部分
* 插入某些查询的结果
```
INSERT INTO 表名
VALUES(...)
```
    其中...表示对应列名的输入
    一次插入必须提供所有列的值，null也算
```
INSERT INTO 表名(列名1，列名2，列名3，...)
VALUES(...);
```
    可以只给某些列赋值
    在这种语法作用下，如果表的结构发生变化，语句依然可以正常生效
```
INSERT INTO 表1(a, b, c, ...)
SELECT a, b, c, ...
FROM 表2
```
    还可以插入检索出的数据，这是唯一一种可以一条语句就插入多行的方法
    但是插入时要注意主键不能重复（如果重复的话，则插入除了主键以外的列组成的行）
```
CREATE TABLE 别名 AS SELECT * FROM 表名
```
    创建一个一模一样的表

1. **任何SELECT选项和子句都可以使用，包括WHERE和GROUP BY**
2. ___可利用联结从多个插入数据___ ~~这个是真他妈难~~
3. 不管数据从哪里来的，最终只能插入到一个表里
# 16 更新和删除数据

## 更新
~~别看了，你有权限吗？~~
1. 更新表中的特定行
2. 更新表中的所有行
```
UPDATE 表名
SET 列1 = 值1,列2 = 值2,...
WHERE 条件
```
    不写WHERE直接更新所有行
**UPDATE语句可以使用子查询，目前还不是很理解**
## 删除
删库的方法非常简单，就像钱一样，花钱的方法多种多样，但是赚钱呢？
```
DELETE FROM 表名
WHERE 条件
```
    不写WHERE就是删除所有行
    DBMS一般不允许删除外键，因为外键通常关联到别的表，在没解除关系前不允许删除外键
# 17 创建和操纵表
1. 使用交互式创建和管理数据库表的工具
2. 用SQL语句建表 
```
CREATE TABLE 表名
(
    列1 数据类型(长度/规格) NOT NULL DEFAULT 值/NULL,
    列2 数据类型(长度/规格) NOT NULL DEFAULT 值/NULL,
);
```
    如果不指定NOT NULL那就是NULL
    数据类型右CHAR INTEGER DECIMAL VARCHAR
你还可以更新表
```
ALTER TABLE 表名
ADD 列名 数据类型()

ALTER TABLE 表名
DROP COLUMN 列名
```
    使用ALTER TABLE前要极为小心，应该在改动前做完整的备份

# 以下的所有部分建议看书
# 18 使用视图
# 19 使用存储过程
# 20 控制事务处理
# 21 使用游标


# presentation

## 演说
1. English Learning Experience

    泛话题，可以结合自己经历说一下怎么学英语的
    
    1. 可以结合歌曲、游戏、文学、影视，从一个中心话题可以扩充到不同角度
    2. 自由度大，更强调个人，小组性弱
2. My fav 系列

    1. 我喜欢的、我最喜欢的XX系列
    2. 介绍具体的某一事物，不容易流水账