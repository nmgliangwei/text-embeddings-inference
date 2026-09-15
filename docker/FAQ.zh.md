
# FAQ

## 1 启动容器失败，报错：Available device not found

可能原因：

- 宿主机安装驱动时未配置--install-for-all参数，容器以普通用户(HwHiAiUser)运行,导致容器内无法访问npu卡

- 指定运行的卡已被其他容器占用

定位解决思路：

- 执行如下命令定位

```bash
# 启动容器
export MODEL_DIR="/home/HwHiAiUser/model"
docker run \
    -u <user> \
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
    -p <host_port>:8000 \
    --entrypoint bash \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag>

# 进入容器
docker exec -it tei_container bash

# 执行如下命令测试是否可以正常查看npu卡信息
npu-smi info
```

## 2 启动容器失败，报错："Error: Could not create backend"

参考 [FAQ 1](#1-启动容器失败报错available-device-not-found) 运行并进入容器

在容器内执行如下命令：

```bash
# 切换到后端目录：
cd /home/HwHiAiUser/text-embeddings-inference/backends/python/server

# 执行后端启动程序，查看是否出现堆栈异常
python3 text_embeddings_server/cli.py $MODEL_DIR/bge-large-zh-v1.5(根据具体模型名修改)

#如果日志出现 "Server started at unix://tmp/text-embeddings-server"表示后端服务启动成功
```

## 3 大并发时npu显存持续增加或存在OOM问题

- **方案1 (推荐)**

服务启动时最小化配置 `-e MAX_CONCURRENT_REQUESTS`值，例如配置为64，限制最大并发访问请求数, 具体[参考链接](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)

- **方案2**

容器启动时可通过配置环境变量`-e CLEAN_NPU_CACHE=True`,使能每次推理后清理npu缓存，注意开启该功能可能会影响推理性能。

- **方案3**

服务启动时配置`-e MAX_BATCH_TOKENS`环境变量进行最大token限制，具体[参考链接](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)

## 4 启动多个容器时，所有容器都在第一张卡上推理

可能原因，容器启动时加了特权选项`--privileged=true`后所有容器都运行在第一张卡上
解放方案：

- 容器启动时只挂载指定的卡，例如`--device=/dev/davinci1`
- 容器启动添加环境变量限制运行的卡，例如 `-e ASCEND_RT_VISIBLE_DEVICES=1`

## 5 期望多个容器运行在一张卡上

- **方案1**

启动容器添加`-u root --privileged=true -e ASCEND_RT_VISIBLE_DEVICES=指定卡号`，注意由于加了特权后，所有的卡都会被挂载到容器中

- **方案2**

参考 [FAQ 1](#1-启动容器失败报错available-device-not-found) 运行并进入容器

在容器内执行如下命令：

```bash
bash start.sh --model-id <模型1绝对路径>  --port <listen port1> &
bash start.sh --model-id <模型2绝对路径>  --port <listen port2> &
```

## 6 启动容器失败，报错："failed to create HTTP listener: Address already in use (os error 98)"

环境上存在多个tei容器，并且使用了主机网络，容器启动时会探测 PROMETHEUS_PORT，默认在9000，容器启动命令中配置此环境变量为其他端口即可
解决方案：

- 容器启动时分配其他端口, 例如：`-e PROMETHEUS_PORT=9001`

## 7 启动容器失败，报错："`--max-batch-tokens` cannot be lower than the model `max_input_length`"

运行容器是报错如下：

Error: `--max-batch-tokens` cannot be lower than the model `max_input_length` (32768 < 16384) when `--auto-truncate` is disabled

解决方案：容器启动时配置 -e AUTO_TRUNCATE=True 开启自动截断, 具体[参考链接](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)

## 8 容器内服务启动失败，需要添加 `--security-opt="seccomp=unconfined"` 参数

现象：按文档命令启动容器后，容器内 TEI 服务启动失败或进程直接退出，报错可能为 `Operation not permitted`、`Permission denied`，或未见明显报错即退出（如 glibc/Python 多线程调用被默认 seccomp 配置拦截的 `clone3` 等系统调用）。

可能原因：Docker 默认会给容器启用 seccomp 安全配置，屏蔽部分系统调用；TEI 依赖的 CANN 驱动栈、Python/torch_npu 后端初始化时会调用这些被屏蔽的系统调用，导致服务启动失败。该问题在 Docker/containerd 版本升级、CANN 版本变化，或从 `--privileged` 迁移到 `--device` 挂载方式后更容易出现。

解决方案：容器启动命令中添加 `--security-opt="seccomp=unconfined"` 关闭默认 seccomp 配置：

```bash
docker run \
    -u <user> \
    --security-opt="seccomp=unconfined" \
    --device /dev/davinci0 \
    ...(其余参数与[快速开始](OVERVIEW.zh.md#运行-tei-容器)中的启动命令一致)
```

注意事项：

- 该参数仅关闭系统调用过滤，不会授予设备访问等额外权限，仍建议通过 `--device` 精确挂载所需卡，不要与 `--privileged` 混用。
- 与 `--privileged` 的区别：`--privileged` 会隐式关闭 seccomp 并授予容器几乎所有主机权限，且会导致多容器运行在第一张卡上（见 [FAQ 4](#4-启动多个容器时所有容器都在第一张卡上推理)）；`--security-opt="seccomp=unconfined"` 只关闭系统调用过滤，影响面更小。
