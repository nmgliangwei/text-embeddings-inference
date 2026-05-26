
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

参考[FAQ 1](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/FAQ.zh.md#L16-L36) 运行并进入容器

在容器内执行如下命令：

```bash
# 切换到后端目录：
cd /home/HwHiAiUser/text-embeddings-inference/backends/python/server

# 执行后端启动程序，查看是否出现堆栈异常
python3 text_embeddings_server/cli.py $MODEL_DIR/bge-large-zh-v1.5(根据具体模型名修改)

#如果日志出现 "Server started at unix://tmp/text-embeddings-server"表示后端服务启动成功
```

## 3 大并发时npu显存持续增加

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

参考[FAQ 1](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/FAQ.zh.md#L16-L36) 运行并进入容器

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
