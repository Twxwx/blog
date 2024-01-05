---
title: Hexo使用指南
date: 2023-12-12 12:28:19
tags:
---


## hexo基本运行指令

``` bash
$ hexo new "article name"  # 新建文章
$ hexo clean  # 清理缓存
$ hexo g      # 生成静态文件
$ hexo s      # 本地预览
$ hexo d      # 部署上线

$ hexo clean && hexo g && hexo d # 一键部署
```

### 图片文件存储
存到themes/source/img

### 浏览器清理缓存并刷新
``` bash 
command + shift + R
```

## markdown操作指令

### 标题

``` markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```
