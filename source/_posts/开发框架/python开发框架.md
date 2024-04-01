---
title: Python开发框架
date: 2024-04-01 11:28:48
categories:
    - 开发框架
tags:
---

## 简介

- FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 3.8+ 并基于标准的 Python 类型提示。

## 安装教程

- 安装fastapi

```python
pip install fastapi
```

- 安装uvicorn：Uvicorn是一个基于ASGI（Asynchronous Server Gateway Interface）的异步Web服务器，用于运行异步Python web应用程序。它是由编写FastAPI框架的开发者设计的，旨在提供高性能和低延迟的Web服务

```python
pip install uvicorn
```

## 快速入门

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.get("/info")
async def info():
    return {
        "app_name": "FastAPI框架学习",
        "app_version": "v0.0.1"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- def：使用 def 关键字定义的函数是同步函数。这意味着当你调用这样的函数时，程序会等待该函数执行完成，并返回结果后，才会继续执行后续的代码。
- async def：使用 async def 定义的函数是异步函数，也称为"协程"。这意味着你可以在该函数中执行异步操作，并且不会阻塞程序的其他部分。

- FastApi框架在启动时，除了注册路由之外，还会自动生成API在线文档,并且生成两种在线文档: Swagger UI 和 ReDoc，访问地址分别为: SwaggerUi风格文档:http://127.0.0.1:8000/docs、ReDoc风格文档：http://127.0.0.1:8000/redoc

