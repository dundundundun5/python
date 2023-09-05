# SQL必知必会
重点学习MYSQL，SQL不区分大小写，注意保持代码可读性
mysql 5和8的区别
<https://blog.csdn.net/Da_Xiong000/article/details/125065335>
```//使用mysql数据库
 USE mysql;

//更改root账户的加密规则为 mysql_native_password 并修改密码为 password 即可
 ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';
 
 //刷新权限
 FLUSH PRIVILEGES;
```

## show指令
1. show tables或show tables from database_name; -- 显示当前数据库中所有表的名称。 
2. show databases; -- 显示mysql中所有数据库的名称。 
3. show columns from table_name from database_name; 或show columns from database_name.table_name; -- 显示表中列名称。 
4. show grants for user_name; -- 显示一个用户的权限，显示结果类似于grant 命令。 
5. show index from table_name; -- 显示表的索引。 
6. show status; -- 显示一些系统特定资源的信息，例如，正在运行的线程数量。 
7. show variables; -- 显示系统变量的名称和值。 
8. show processlist; -- 显示系统中正在运行的所有进程，也就是当前正在执行的查询。大多数用户可以查看他们自己的进程，但是如果他们拥有process权限，就可以查看所有人的进程，包括密码。 
9. show table status; -- 显示当前使用或者指定的database中的每个表的信息。信息包括表类型和表的最新更新时间。 
10. show privileges; -- 显示服务器所支持的不同权限。 
11. show create database database_name; -- 显示create database 语句是否能够创建指定的数据库。 
12. show create table table_name; -- 显示create database 语句是否能够创建指定的数据库。 
13. show engines; -- 显示安装以后可用的存储引擎和默认引擎。 
14. show innodb status; -- 显示innoDB存储引擎的状态。 
15. show logs; -- 显示BDB存储引擎的日志。 
16. show warnings; -- 显示最后一个执行的语句所产生的错误、警告和通知。 
17. show errors; -- 只显示最后一个执行语句所产生的错误。 
18. show [storage] engines; --显示安装后的可用存储引擎和默认引擎。
## 数据库设计规范
1. 1NF

    第一范式要求表内所有内容不可再分，（姓名，家庭联系人）就不是第一范式，一定要具体到某个属性
2. 2NF

    要求表里有一个主键，其他属性必须依赖于这个，可以同名同姓，但是学号一定不同
3. 3NF

    消除传递依赖的情况是第三范式

    一个表里，A属性依赖于B属性，结果B属性又依赖于C属性，这样的话就要继续拆分表了
4. BCNF

    是第三范式的补充

    意思是表中出现两个或多个同时具有主键功能的字段，但是表的主键只能有一个。那么需要单独存一张或几张表，来表示这些“小主键”之间的对应关系
## 数据库三级模式
1. 模式：概念模式、逻辑模式
3. 外模式：用户模式
4. 内模式：物理模式
## SQL数据类型
1. 字符串
   
   1. char(n) 会用空格填充
   2. varchar(n) 不会用空格填充
2. 数字

    1. smallint (-32768,32767)
    2. int (-2147483648,2147483647)
    3. bigint (-9223372036854775808,9223372036854775807)
    4. float 单精度浮点数
    5. double 双精度浮点数
3. 时间
   
    1. date 日期
    2. time 时间
    3. year 年份
    4. datetime 年月日时间
    5. timestamp 时间戳
4. 枚举

    enum(值1，值2，等等)
## SQL约束条件
1. 列级约束条件

    1. PRIMARY KEY主键
    2. FOREIGN KEY外键
    3. UNIQUE唯一
    4. CHECK检查
    5. DEFAULT默认值
    6. NOT NULL/NULL非空/空
2. 表级约束条件

    1. 主键
    2. 外键
    3. 唯一
    4. 检查
## SQL用户和权限管理
1. 创建用户

```CREATE USER 用户名 IDENTIFIED BY 密码```

```CREATE USER 用户名```

2. 登录用户

```mysql -u 用户名 -p ```

然后输入密码





3. 用户授权

```grant all|权限1|权限2,...,权限n(列1,列2,...,列n) on 数据库.表 to 用户名 with grant option;```

数据库.表可以用*.*代替

with grant option表示使用户具备授权能力

```revoke all|权限1|权限2,...,权限n(列1,列2,...,列n) on 数据库.表 from 用户名;```

用于收回权限


```grant all privileges on *.* to 用户名@'192.168.%.%' with grant option;```

还能限制登陆用户名的IP地址


最后还要```flush privileges;```

高版本无法使用```grant all privileges on *.* to "root"@"%" identified by "xxxx";```去修改用户权限。
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

__WHERE管的是所有行分组前的通病__

__HAVING管的是按列分组后，分组们的通病__

```
SELECT vend_id, COUNT(*) AS num_prods
FROM Products
WHERE prod_price >= 4
GROUP BY vend_id
HAVING COUNT(*) >= 2;
```
    where先过滤掉所有行中prod_price小于4的行，然后对vend_id进行分组，
    筛除总行数小于2的分组
1. ORDER BY & GROUP BY

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
    AS是表的别名，AS有时可以省略
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

## 15 插入数据
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
## 16 更新和删除数据

### 更新
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
### 删除
删库的方法非常简单，就像钱一样，花钱的方法多种多样，但是赚钱呢？
```
DELETE FROM 表名
WHERE 条件
```
    不写WHERE就是删除所有行
    DBMS一般不允许删除外键，因为外键通常关联到别的表，在没解除关系前不允许删除外键
## 17 创建和操纵
1. 使用交互式创建和管理数据库表的工具
2. 用SQL语句建表 
```
CREATE TABLE 表名
(
    列1 数据类型(长度/规格) NOT NULL DEFAULT 值/NULL,
    列2 数据类型(长度/规格) NOT NULL DEFAULT 值/NULL,
    CONSTRAINT 外键名 FOREIGN KEY (字段名) REFERENCES 数据库名.主表名 (主键列)
); 只有一列是主键，要加上PRIMARY KEY
```
    如果不指定NOT NULL那就是NULL
    数据类型右CHAR INTEGER DECIMAL VARCHAR
你还可以更新表或删除表
```
ALTER TABLE 表名
ADD 列名 数据类型() 

ALTER TABLE 表名
DROP COLUMN 列名[restrict|cascade]

ALTER TABLE 表名
MODIFY 列名 新数据类型 

DROP TABLE 表名[restrict|cascade]
```
    外键列：restrict会让你无法删除带有外键的某一行，而cascade会删除表中所有带有这个外键的行，相当于蒸发
    使用ALTER TABLE前要极为小心，应该在改动前做完整的备份

## 数据库本身的操作
* CREATE DATABASE database_name;
* DROP DATABASE database_name;
* CREATE DATABASE IF NOT EXISTS DATABASE_NAME DEFAULT CHARSET 编码(用utf8) COLLATE 排序规则(用utf8_general_ci);

## 视图
视图（View）是一种虚拟存在的表。视图中的数据并不在数据库中实际存在，行和列数据来自定义视图的查询中使用的表，并且是在使用视图时动态生成的。

视图用于包装SELECT语句，把查询出来的内容当作另一张表看待，可以用update select delete语句修改视图

视图相当于对查询结果的装修，对视图的修改就是对原表的修改

视图可以嵌套
1. 创建视图

```CREATE  [OR REPLACE] VIEW 视图名称[(列名列表)] AS SELECT语句 [WITH[CASCADED | LOCAL] CHECK  OPTION]```

2. 查询视图

```
查看创建视图语句：SHOW CREATE VIEW 

视图名称查看视图数据：SELECT * FROM 视图名称......
```

3. 修改视图

```
方式一：CREATE  [OR REPLACE]  VIEW 视图名称[(列名列表)] AS SELECT语句 [WITH[CASCADED | LOCAL] CHECK OPTION]

方式二：ALTER VIEW 视图名称[(列名列表)] AS SELECT语句 [WITH[CASCADED | LOCAL] CHECK OPTION]
```
4. 删除视图

```DROP VIEW [IF EXISTS] 视图名称 [,视图名称]......```

5. 不允许更新视图的情况

    1. 视图由2个表以上导出
    2. 字段来自聚集函数
    3. 包装的select语句含有group by
    4. select 语句含有DISTINCT
    5. 嵌套查询内层是基本表
    6. 不允许更新的视图的视图
    7. 字段来自字段表达式或常数，只允许delete
## 索引
用额外开销，用于快速查找
* 创建索引

```CREATE INDEX 索引名称 ON 表名/列名```

```SHOW INDEX FROM 表名```

```DROP INDEX 索引名称 ON 表名```

## 触发器
<https://zhuanlan.zhihu.com/p/158670286>

* 触发器（trigger）：监视某种情况，并触发执行某种操作。触发器是在表中数据发生更改时自动触发执行的，它是与表事件相关的特殊的存储过程，它的执行不是由程序调用，也不是手工启动，而是由事件来触发，例如当对一个表进行操作（insert，delete， update）时就会激活它执行。也就是说触发器只执行DML事件(insert、update和delete)

```
CREATE TRIGGER <trigger_name>  
  
BEFORE|AFTER
 
INSERT|UPDATE|DELETE  ON <table_name> # 表名
  
FOR EACH ROW  # 这句话在mysql是固定的
 
BEGIN

<触发的SQL语句>（调用NEW/OLD参数）;
 
END
```

```show triggers [\G]```用于显示触发器,\G是显示格式

1.CREATE TRIGGER <trigger_name> --- 触发器必须有名字，最多64个字符，可能后面会附有分隔符.它和MySQL中其他对象的命名方式基本相象.

2.{ BEFORE | AFTER } --- 触发器触发时间设置：可以设置为事件发生前或后（前：一般用于校验；后：一般用于关联）。

3.{ INSERT | UPDATE | DELETE } -- 设定触发事件：如执行insert、update或delete的过程时激活触发器。

4.ON <table_name> --- 触发器是属于某一个表的: 当在这个表上执行 INSERT|UPDATE|DELETE 操作的时候就导致触发器的激活. 同时，我们不能给同一张表的“同一个事件”安排两个触发器（意味着不能同时有两个Insert触发器）。

5.FOR EACH ROW --- 触发器的执行间隔（必有的公式内容）：FOR EACH ROW子句通知触发器 每隔一行执行一次动作，而不是对整个表执行一次。

6.<触发的SQL语句> --- 触发器包含所要触发的SQL语句：这里的语句可以是任何合法的语句， 包括复合语句，但是这里的语句受的限制和函数的一样。当然，触发SQL中可以调用“触发了（ INSERT | UPDATE | DELETE ）触发器的那一行数据”

* 在INSERT型触发器中，NEW用来表示将要（BEFORE）或已经（AFTER）插入的新数据；
* 在UPDATE型触发器中，OLD用来表示将要或已经被修改的原数据，NEW用来表示将要或已经修改为的新数据；
* 在DELETE型触发器中，OLD用来表示将要或已经被删除的原数据；

## 事务
<https://blog.csdn.net/qq_56880706/article/details/122653735>
```show engines```查看存储引擎，innodb支持事务

往通俗的讲就是，事务就是一个整体，里面的内容要么都执行成功，要么都不成功。不可能存在部分执行成功而部分执行不成功的情况。

* 原子性（Atomicity）：不可被中途打断，否则全不执行
* 一致性（Consistency）：事务必须使数据库从一个一致状态变换到另外一个一致状态
* 隔离性（Isolation）：一个事务的执行不能被其他事务干扰。读未提交、读提交、可重复读、串行化
* 持久性（Durability）：一个事务一旦提交成功，它对数据库中数据的改变将是永久性的

```
begin;
SQL语句
rollback; # 回滚事务
savepoint 存档点;
rollback to 存档点;

commit; # 提交事务 一旦提交就无法回滚了
```

禁用自动提交 ```set autocommit = 0  ```

1. read uncommitted（读未提交数据）：允许事务读取未被其他事务提交的变更。（脏读、不可重复读和幻读的问题都会出现）。
2. read committed（读已提交数据）：只允许事务读取已经被其他事务提交的变更。（可以避免脏读，但不可重复读和幻读的问题仍然可能出现）
3.repeatable read（可重复读）：确保事务可以多次从一个字段中读取相同的值，在这个事务持续期间，禁止其他事务对这个字段进行更新(update)。（可以避免脏读和不可重复读，但幻读仍然存在）
4. serializable（串行化）：确保事务可以从一个表中读取相同的行，在这个事务持续期间，禁止其他事务对该表执行插入、更新和删除操作，所有并发问题都可避免，但性能十分低下（因为你不完成就都不可以弄，效率太低）

## 存储过程
存储过程（Stored Procedure）是一种在数据库中存储复杂程序，以便外部程序调用的一种数据库对象。

存储过程是为了完成特定功能的SQL语句集，经编译创建并保存在数据库中，用户可通过指定存储过程的名字并给定参数(需要时)来调用执行。

1. 关键语法

创建存储过程、函数

```CREATE PROCEDURE/FUNCTION demo_in_parameter(IN p_in int) ```

其他操作

```ALTER/DROP PROCEDURE/FUNCTION```

声明语句结束符，可以自定义：

```DELIMITER $$ 或 DELIMITER //```

存储过程开始和结束

```BEGIN END```

变量赋值

```SET @p_in=1```

变量定义

```DECLARE l_int int unsigned default 4000000; ```