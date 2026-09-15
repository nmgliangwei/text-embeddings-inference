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

该镜像基于text embedding inference 适配华为NPU卡提供embedding、reranker等服务化接口，接口兼容开源text embedding inference, 支持embedding和reranker、Sequence Classification等模型，该镜像默认以普通用户`HwHiAiUser`用户运行


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

### 镜像配套表
| TEI    | CANN  | torch-npu |
|--------|-------|-----------|
| 26.0.0 | 9.0.0 | 2.9.0     |

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
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
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
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag>  /home/HwHiAiUser/model/<model_name> <listen_ip> <listen_port>
```

</details>

#### **参数说明**

- `MODEL_DIR` 环境变量表示宿主机上模型存放目录，在运行容器之前需将模型权重文件下载存放此目录下。
- `PYTORCH_NPU_ALLOC_CONF` npu内存块允许切分上限，消息介绍请参考[链接](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/PYTORCH_NPU_ALLOC_CONF.md)
- 请根据实际情况替换 `<user>`、`<tag>`、`<model_name>`、`<listen_ip>`、`<listen_port>`为实际值。
- 请确保宿主机上的模型文件与 `<model_name>` 匹配。
- 容器默认以普通用户`HwHiAiUser`运行, 请确保HwHiAiUser用户有读写宿主机上模型权限，如需指定其他用户，请在运行容器时添加 `-u <user>` 参数。
- （可选）`--security-opt="seccomp=unconfined"`：仅当容器内 TEI 服务启动失败（如报 `Operation not permitted`、`Permission denied` 或进程直接退出）时按需添加。该参数关闭默认 seccomp 安全配置，避免默认 profile 拦截 TEI/CANN 运行所需的系统调用（如 `clone3`）。仅关闭系统调用过滤，不影响 `--device` 设备隔离，可替代 `--privileged` 使用。详见 [FAQ 8](FAQ.zh.md#8-容器内服务启动失败需要添加-security-optseccompunconfined-参数)。
- 请参考 [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md) 了解其他配置参数。

## 支持的模型列表和各模型特定参数说明

### 文本嵌入模型
| 模型名称                                                                                                    | 模型类型        | 模型启动必配参数                                       |
|---------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| [BAAI/bge-large-zh-v1.5](https://www.modelscope.cn/models/BAAI/bge-large-zh-v1.5)                       | BERT        | -                                              |
| [BAAI/bge-m3](https://www.modelscope.cn/models/BAAI/bge-m3)                                             | XLM-RoBERTa | -                                              |
| [aspire/acge_text_embedding](https://www.modelscope.cn/models/yangjhchs/acge_text_embedding)            | BERT        | -                                              |
| [jinaai/jina-embeddings-v2-base-zh](https://www.modelscope.cn/models/jinaai/jina-embeddings-v2-base-zh) | BERT        | -e TRUST_REMOTE_CODE=1                         |
| [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://www.modelscope.cn/models/iic/gte-Qwen2-1.5B-instruct)     | Qwen2       | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false |
| [Alibaba-NLP/gte-Qwen2-7B-instruct](https://www.modelscope.cn/models/iic/gte-Qwen2-7B-instruct)         | Qwen2       | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false |
| [Qwen/Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)                 | Qwen3       | -                                              |
| [Qwen/Qwen3-Embedding-4B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B)                     | Qwen3       | -                                              |
| [Qwen/Qwen3-Embedding-8B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B)                     | Qwen3       | -                                              |
| [microsoft/unixcoder-base](https://www.modelscope.cn/models/microsoft/unixcoder-base)                   | RoBERTa     | -                                              |


### 序列分类和排序模型
| 模型名称                                                                                        | 模型类型        | 模型启动必配参数                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [BAAI/bge-reranker-large](https://www.modelscope.cn/models/BAAI/bge-reranker-large)         | XLM-RoBERTa | -                                                                                                                                                                            |
| [BAAI/bge-reranker-v2-m3](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3)         | XLM-RoBERTa | -                                                                                                                                                                            |
| [Qwen/Qwen3-Reranker-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B)       | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [Qwen/Qwen3-Reranker-4B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B)           | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [Qwen/Qwen3-Reranker-8B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B)           | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [jason9693/Qwen2.5-1.5B-apeach](https://www.modelscope.cn/models/Howeee/Qwen2.5-1.5B-apeac) | Qwen2       | -                                                                                                                                                                            |


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
