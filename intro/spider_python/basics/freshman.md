# 入门
## 常用方法

1. urlopen()
表示向网站发起请求并获取响应对象，如下所示：
        
        urllib.request.urlopen(url,timeout)

    urlopen() 有两个参数，说明如下：
        
        url：表示要爬取数据的 url 地址。
        timeout：设置等待超时时间，指定时间内未得到响应则抛出超时异常。
2. Request()

该方法用于创建请求对象、包装请求头，比如 ___重构 User-Agent___（即用户代理，指用户使用的浏览器）使程序更像人类的请求，而非机器。重构 User-Agent 是爬虫和反爬虫斗争的第一步。

* User-Agent

    User-Agent 即用户代理，简称“UA”，它是一个特殊字符串头。网站服务器通过识别 “UA”来确定用户所使用的操作系统版本、CPU 类型、浏览器版本等信息。而网站服务器则通过判断 UA 来给客户端发送不同的页面。
* User-Agent汇总

    |系统|浏览器|字符串|
    |-|-|-|
    |Windows|Chrome|	 Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36|
    |IOS|Chrome| Mozilla/5.0 (iPhone; CPU iPhone OS 7_0_4 like Mac OS X) AppleWebKit/537.51.1 (KHTML, like Gecko) CriOS/31.0.1650.18 Mobile/11B554a Safari/8536.25|
    |IOS|Safari|Mozilla/5.0 (iPhone; CPU iPhone OS 8_3 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12F70 Safari/600.1.4|
* 构建UA池

    在python中属于字符串列表，每个字符串都是一个浏览器的UA

    人工获取UA比较复杂，还可以用fake-useragent获取伪装的UA
3. 
## 基础概念

1. URL组成

    URL 是由一些简单的组件构成，比如协议、域名、端口号、路径和查询字符串等

            https://www.youtube.com/watch?v=nrRj9J7dFlY&t=1783s 
   * :  __分隔协议和主机组件__
   * ?  __分隔路径和查询参数等__
   * =  __表示查询参数中的键值对__
   * \+ __表示空格__
   * &  __分隔查询多个键值对__
   * % __指定特殊字符__
   * \# __表示书签__
   * / __分隔目录和子目录__
2. URL中需要编码的字符

    1. ASCII 表中没有对应的可显示字符，例如，汉字。
    2. 不安全字符，包括：# ”% <> [] {} | \ ^ ` 。
    3. 部分保留字符，即 & / : ; = ? @ 。