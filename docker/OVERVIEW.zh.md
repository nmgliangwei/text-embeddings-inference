# 昇腾 text-embeddings-inference(TEI)

> [English](./OVERVIEW.md) | 中文

## 快速参考
- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/text-embeddings-inference/issues)
  - [Ascend/TEI 代码](https://gitcode.com/Ascend/text-embeddings-inference)
  - [AscendHub镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)
  - [Ascend社区](https://www.hiascend.com/)

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式：

```bash
<TEI版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段         | 示例值                            | 说明        |
|------------|--------------------------------|-----------|
| `TEI版本`    | `26.0.0`                       | TEI 版本号   |
| `芯片系列`     | `910b`、`a3`、`310p`             | 目标昇腾芯片系列  |
| `操作系统`     | `ubuntu22.04`、`openeuler24.03` | 基础操作系统    |
| `python版本` | `py3.11`                       | Python 版本 |



### CANN 9.0.0 + 26.0.0 TEI镜像

| Tag                                 | Dockerfile                                | Image Content |
|-------------------------------------|-------------------------------------------|---------------|
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](./Dockerfile.910b.ubuntu)    | cann + TEI    |
| `26.0.0-910b-openeuler24.03-py3.11` | [Dockerfile](./Dockerfile.910b.openeuler) | cann + TEI    |
| `26.0.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](./Dockerfile.a3.ubuntu)      | cann + TEI    |
| `26.0.0-a3-openeuler24.03-py3.11`   | [Dockerfile](./Dockerfile.a3.openeuler)   | cann + TEI    |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](./Dockerfile.310p.ubuntu)    | cann + TEI    |
| `26.0.0-310p-openeuler24.03-py3.11` | [Dockerfile](./Dockerfile.310p.openeuler) | cann + TEI    |

---

## 快速开始

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的昇腾 NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 运行 TEI 容器

```bash
# 配置宿主机模型存放目录
export MODEL_DIR="/home/HwHiAiUser/model"
# 根据实际情况替换 <tag> 为实际 tag, <model name> 为实际模型名称, <ip> 为实际 IP
docker run \
    --name tei_container \
    --device /dev/davinci1 \
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
    -p 8000:8000 \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag> --model-id $MODEL_DIR/<model name> --hostname <ip> --port 8000
```

### 如何构建容器镜像

```bash
# 在宿主机上任意目录下 clone 代码, 并进入 docker 目录, 获取dockerfile替换 {cann version} 为实际版本, 替换 {your_repo} 为实际仓库
git clone https://gitcode.com/Ascend/text-embeddings-inference.git && cd text-embeddings-inference/docker

docker build --build-arg CANN_VERSION={cann version} -t {your_repo}/mis-tei:latest -f Dockerfile.<芯片系列>.<操作系统> .
```

---

## 支持的硬件

| 芯片系列    | 产品示例                          | 架构             |
|---------|-------------------------------|----------------|
| 昇腾 910B | Atlas 800I A2                 | ARM64 / x86_64 |
| 昇腾 A3   | Atlas 800I A3                 | ARM64 / x86_64 |
| 昇腾 310P | Atlas 300I Pro、Atlas 300V Pro | ARM64 / x86_64 |

---

## 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
