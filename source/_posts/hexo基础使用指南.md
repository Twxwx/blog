---
title: Hexo使用指南
date: 2023-12-12 12:28:19
categories: 
    - 指令
---

[butterfly主题文档](https://butterfly.js.org/posts/21cfbf15/)

## hexo基本运行指令

``` bash
$ hexo clean  # 清理缓存
$ hexo g      # 生成网站的静态文件到默认的设置文件public中
$ hexo s      # 本地预览
$ hexo d      # 自动生成网站静态文件，并部署到设定的仓库中

$ hexo clean && hexo g && hexo s # 本地预览
$ hexo clean && hexo g && hexo d # 一键部署

将文章的md文件删除，如果文章已经发布，那么还需要将.deploy_git也给删除
```

### 新建文章
``` bash
$ hexo new "article name"  # 新建文章

文章基本设置
---
title: article name
date: 2017-12-02 21:01:24
categories:  #分类
    - xxx
tags:   #标签
    - xx
    - xx
---
```

### 图片文件存储及使用
1. 将图片存到themes/butterfly/source/img
    - 如果要使用github上的图片，将链接里的blob改为raw 
    - 如果要使用github上的图片，直接用相对路径 /img/xxx.jpg

## git的一些操作
``` bash
git clone {url} # 从github克隆项目
git remote -v # 查看项目关联的远程仓库
git add . # 将代码存入暂存区
git commit -m {message} # 提交代码
git push # 推送代码到远程仓库
git pull # 拉取远程代码到本地
git checkout {branch} # 切换分支
git merge {branch} # 合并分支
```


## 浏览器清理缓存并刷新
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

![图片](/img/index_img.png)