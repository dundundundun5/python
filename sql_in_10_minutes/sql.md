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

## 12 联结表
由于没有学过关系表和关系数据库

需要先补充 关系表和关系数据库

关系表<https://blog.csdn.net/YXXXYX/article/details/123270424>

ER图<https://blog.csdn.net/caohongxing/article/details/122398825>

