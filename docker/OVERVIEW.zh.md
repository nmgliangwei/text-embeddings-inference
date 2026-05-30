# text-embeddings-inference(TEI)

> [English](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/OVERVIEW.md) | 中文

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/text-embeddings-inference/issues)
  - [text-embeddings-inference 代码](https://gitcode.com/Ascend/text-embeddings-inference)
  - [text-embeddings-inference API Reference](https://huggingface.github.io/text-embeddings-inference/)
  - [镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)
  - [社区](https://www.hiascend.com/)

## 概述

该镜像基于text embedding inference 适配华为NPU卡提供embedding、reranker等服务化接口，接口兼容开源text embedding inference, 支持embedding和reranker、Sequence Classification等模型，该镜像默认以普通用户HwHiAiUser用户运行


## 镜像Tags对应的Dockerfile 链接

### Tag 规范

Tag 遵循以下格式：

```bash
<TEI版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段         | 示例值                            | 说明        |
|------------|--------------------------------|-----------|
| `TEI版本`    | `26.0.0`                       | TEI 版本号   |
| `芯片系列`     | `910`, `a3`, `Atlas 300I Pro`  | 目标芯片系列    |
| `操作系统`     | `ubuntu22.04`、`openeuler24.03` | 基础操作系统    |
| `python版本` | `py3.11`                       | Python 版本 |

### CANN 9.0.0 + 26.0.0 TEI镜像

| Tag                                 | Dockerfile                                                                                                    | Image Content |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------|---------------|
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.910b.ubuntu)    | cann + TEI    |
| `26.0.0-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.910b.openeuler) | cann + TEI    |
| `26.0.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.a3.ubuntu)      | cann + TEI    |
| `26.0.0-a3-openeuler24.03-py3.11`   | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.a3.openeuler)   | cann + TEI    |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.310p.ubuntu)    | cann + TEI    |
| `26.0.0-310p-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.310p.openeuler) | cann + TEI    |

---

## 快速开始

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的NPU 驱动, 请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 运行 TEI 容器

26.0.0及以后的版本启动命令:

```bash
export MODEL_DIR=/home/HwHiAiUser/model
docker run \
    -u <user> \
    --net host \
    --name tei_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -e MODEL_DIR=$MODEL_DIR \
    -v $MODEL_DIR:$MODEL_DIR \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag> --model-id $MODEL_DIR/<model_name> --hostname <listen_ip> --port <listen_port>
```

<details> <summary>26.0.0之前的版本启动命令:</summary>

```bash
export MODEL_DIR=/home/HwHiAiUser/model
docker run \
    -u <user> \
    --net host \
    --name tei_container \
    -e ENABLE_BOOST=True \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $MODEL_DIR:/home/HwHiAiUser/model \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag>  /home/HwHiAiUser/model/<model_name> <listen_ip> <listen_port>
```

</details>

#### **参数说明**

- `MODEL_DIR` 环境变量指定了宿主机上的模型存放目录，容器内 TEI 会从该目录加载模型。
- 请根据实际情况替换 `<user>`、`<tag>`、`<model_name>`、`<listen_ip>`、`<listen_port>`为实际值。
- 请确保宿主机上的模型文件与 `<model_name>` 匹配。
- 容器默认以普通用户`HwHiAiUser`运行, 请确保HwHiAiUser用户有读写宿主机上模型权限，如需指定其他用户，请在运行容器时添加 `-u <user>` 参数。
- 请参考 [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md) 了解其他配置参数。

## 支持的模型列表和各模型特定参数说明

| 模型名                                 | 说明        | 其他必配参数                                                                                                                                                                       |
|-------------------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BAAI/bge-large-zh-v1.5              | 稠密向量模型    | -                                                                                                                                                                            |
| BAAI/bge-m3                         | 稠密和稀疏向量模型 | -                                                                                                                                                                            |
| aspire/acge_text_embedding          | 稠密向量模型    | -                                                                                                                                                                            |
| BAAI/bge-reranker-large             | 排序模型      | -                                                                                                                                                                            |
| BAAI/bge-reranker-v2-m3             | 排序模型      | -                                                                                                                                                                            |
| jinaai/jina-embeddings-v2-base-zh   | 稠密向量模型    | -e TRUST_REMOTE_CODE=1                                                                                                                                                       |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 稠密向量模型    | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false                                                                                                                               |
| Alibaba-NLP/gte-Qwen2-7B-instruct   | 稠密向量模型    | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false                                                                                                                               |
| Qwen/Qwen3-Embedding-0.6B(4B、8B)    | 稠密向量模型    | -                                                                                                                                                                            |
| Qwen/Qwen3-Reranker-0.6B(4B、8B)     | 排序模型      | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| microsoft/unixcoder-base            | 稠密向量模型    | -                                                                                                                                                                            |

## 接口测试样例

可通过执行如下命令测试各模型服务功能是否正常运行

- **排序模型测试**:

```bash
curl 127.0.0.1:8080/v1/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "documents": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

- **稠密向量模型测试**:

```bash
curl 127.0.0.1:8080/v1/embeddings \
    -X POST \
    -d '{"input":["The capital of China is Beijing."]}' \
    -H 'Content-Type: application/json'
```

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["The capital of China is Beijing."]}' \
    -H 'Content-Type: application/json'
```

- **序列分类模型测试**

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

- **稀疏向量模型测试**

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### 如何构建容器镜像

```bash
# 在宿主机上任意目录下 clone 代码, 并进入 docker 目录, 获取dockerfile替换 {cann version} 为实际版本, 替换 {your_repo} 为实际镜像仓库
git clone https://gitcode.com/Ascend/text-embeddings-inference.git && cd text-embeddings-inference/docker

docker build --network host --build-arg CANN_VERSION={cann version} -t {your_repo}/mis-tei:latest -f Dockerfile.<芯片系列>.<操作系统> .
```

---

## 支持的硬件

| 芯片系列           | 产品示例                           | 架构             |
|----------------|--------------------------------|----------------|
| Atlas 910      | Atlas 800I A2                  | ARM64 / x86_64 |
| Atlas A3       | Atlas 800I A3                  | ARM64 / x86_64 |
| Atlas 300I Pro | Atlas 300I Pro, Atlas 300V Pro | ARM64 / x86_64 |

---

## 故障排除

请先阅读文档的[常见问题](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/FAQ.zh.md)部分，如果问题未解决，请联系[社区](https://gitcode.com/Ascend/text-embeddings-inference/issues)获取帮助。

## 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
