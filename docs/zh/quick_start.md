# 快速入门

本教程指导使用TEI镜像部署BAAI/bge-large-zh-v1.5模型，帮助您快速掌握TEI部署和使用。

## 步骤 1：拉取镜像

1. **确定待下载镜像版本**
   - 访问昇腾社区[镜像仓](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)获取镜像最新版本以及与之配套的CANN版本
   - 根据当前硬件型号（如 Atlas 800I A2 推理服务器）选择对应版本
   > [!NOTE]
   >
   > - 镜像中已安装CANN，无需重复安装
   > - 注意区分 CPU 架构（x86_64 / aarch64）

2. **环境预检查**
   - 执行 `npu-smi info` 命令获取 NPU 驱动版本
   - 通过CANN版本确定[固件与驱动配套](https://www.hiascend.com/hardware/firmware-drivers/community)，如果驱动版本不配套，请参考[固件与驱动文档](https://www.hiascend.com/hardware/firmware-drivers/community)更新NPU驱动。

3. **下载镜像**
   镜像 Tag 格式为 `{version}-{chip}-{os}-{python}-{arch}`，各变量含义如下：

 | 字段         | 示例值                            | 说明        |
|------------|--------------------------------|-----------|
| `TEI版本`    | `26.0.0`                       | TEI 版本号   |
| `芯片系列`     | `910`, `a3`, `Atlas 300I Pro`  | 目标芯片系列    |
| `操作系统`     | `ubuntu22.04`、`openeuler24.03` | 基础操作系统    |
| `python版本` | `py3.11`                       | Python 版本 |

首先确保您的环境存在 NPU 设备，并已安装 NPU 驱动和固件,下载和当前硬件配套的镜像版本[TEI镜像](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d),这里以以 26.0.0 版本、910b 芯片、Ubuntu 22.04、Python 3.11 架构为例。
```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11
```
## 步骤 2： 下载模型权重
下载模型权重到/home/data目录下,这里以bge-large-zh-v1.5模型为例。

```bash
   cd /home/data
   GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/BAAI/bge-large-zh-v1.5.git
   cd bge-large-zh-v1.5 && git lfs pull
 ```
## 步骤 3： 运行容器

```bash
    # 配置模型权重存放目录
    export MODEL_DIR=/home/data
    docker run -itd \
    -u root \
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
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11 --model-id $MODEL_DIR/bge-large-zh-v1.5 --hostname 127.0.0.1 --port 8000
  ```

   - 可以通过 `ls -al /dev/davinci*` 命令检查可以挂载的设备文件
   - 容器启动命令参数请参考 [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md) 。

## 步骤 4： 服务请求测试
访问服务,获得向量结果
   ```bash
   curl 127.0.0.1:8080/v1/embeddings \
       -X POST \
       -d '{"input":["The capital of China is Beijing."]}' \
       -H 'Content-Type: application/json'
   ```
