# 贡献

感谢您考虑为 TEI 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

您可以通过多种方式支持本项目：

- 通过[Issues](https://gitcode.com/Ascend/text-embeddings-inference/issues)反馈问题。
- 建议或实现新功能。
- 改进或扩展文档。
- 审查Pull Request并协助其他贡献者。
- 传播项目：在博客文章、社交媒体上分享 TEI，或给仓库点个⭐。

请先提前了解社区相关规范：

- [Ascend开源项目行为守则](https://gitcode.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)
- [Issue提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [Issue处理流程说明](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [Ascend社区开发者测试贡献指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend开源与第三方软件建仓及分支命名指导](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend开源与第三方软件管理规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [社区安全设计规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)
- [代码规范](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- [PR提交指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)

## 开发与测试
1. **开发环境准备**

   下载容器镜像,以下载TEI 26.0.0-910b-ubuntu22.04-py3.11版本为例:
   ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11
    ```
   运行容器：
   ```bash
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
        --entrypoint bash \
        -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11
    ```
   登录容器：
   ```bash
   docker exec -it tei_container bash
   ```
2. **Fork 仓库** <br>
在GitCode平台代码仓库右上角点击"Fork"按钮，Fork一份源代码到个人仓
3. **克隆代码**<br>
在容器内克隆个人仓代码并开发
   ```bash
   git clone https://gitcode.com/<your-username>/text-embeddings-inference.git
   ```
4. **编译代码**
   ```bash
   export PATH="/home/HwHiAiUser/.cargo/bin:$PATH"
   CARGO_BUILD_RUSTFLAGS="-C target-cpu=generic" bash -c "cargo install --path router -F python -F http --no-default-features"
   ```
5. **验证功能**
以部署bge-large-zh-v1.5模型为例
   ```bash
    bash start.sh --model-id $MODEL_DIR/bge-large-zh-v1.5 --hostname 127.0.0.1 --port 8000
   ```
   - 访问服务，测试服务功能是否符合预期
   ```bash
   curl 127.0.0.1:8080/v1/embeddings \
       -X POST \
       -d '{"input":["The capital of China is Beijing."]}' \
       -H 'Content-Type: application/json'
   ```
6. **补充测试用例**
   在tests目录下添加测试用例，测试新增的功能是否符合预期，如果需要执行全量用例测试，需在`MODEL_DIR`环境变量目录下准备所有用例模型权重
   ```bash
      export MODEL_DIR=/home/data #根据模型权重目录适配修改
      bash run_presmoke.sh
   ```
7. **执行pre-commit检查**

   本地提交代码前请先执行pre-commit检查，检查指导参见[pre-commit本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

8. **提交Pull Request**

   提交PR并等待代码审查

如有任何疑问或需要帮助，请随时在[Issues](https://gitcode.com/Ascend/text-embeddings-inference/issues)中提问，或通过其他社区渠道与我们联系。
