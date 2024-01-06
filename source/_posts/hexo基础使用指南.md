---
title: Hexo使用指南
date: 2023-12-12 12:28:19
tags:
---


## hexo基本运行指令

``` bash
$ hexo new "article name"  # 新建文章
$ hexo clean  # 清理缓存
$ hexo g      # 生成网站的静态文件到默认的设置文件public中
$ hexo s      # 本地预览
$ hexo d      # 自动生成网站静态文件，并部署到设定的仓库中

$ hexo clean && hexo g && hexo s # 本地预览
$ hexo clean && hexo g && hexo d # 一键部署
```

### 图片文件存储及使用
1. 将图片存到themes/butterfly/source/img
2.1 如果要使用github上的图片，将链接里的blob改为raw 
2.2 如果要使用github上的图片，直接用相对路径 /img/xxx.jpg

### 浏览器清理缓存并刷新
``` bash 
command + shift + R
```

## Markdown操作指令

### 标题

``` markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

### 无序列表
Markdown无序列表写法非常简单，减号（或加号、星号）加一个空格即可。
注：推荐使用减号，因为星号常用于斜粗体。
``` markdown
- Markdown无序列表
+ Markdown无序列表
* Markdown无序列表
```

### 有序列表
Markdown有序列表写法非常简单，数字加小数点，然后加一个空格即可。
``` markdown
1. Markdown有序列表
2. Markdown有序列表
```

### 图片
``` markdown
![图片文字描述](img url)
```

### 链接
``` markdown
法1: <链接>
法2: [文本](链接)
```

### 分割线
``` markdown
* * *
***
*****
- - -
---------------------------------------
```

***

[参考文档](https://markdown.p2hp.com/basic-syntax/)

***

<!-- ![](https://github.com/Twxwx/Twxwx.github.io/raw/master/img/index_img.png) -->
![](/img/index_img.png)