# Contributing

Thank you for considering contributing to TEI! We welcome contributions of all kinds, including bug fixes, feature enhancements, documentation improvements, and even just feedback. Whether you are an experienced developer or a first-time participant in an open-source project, your help is invaluable.

You can support this project in the following ways:

- Report issues through [Issues](https://atomgit.com/Ascend/text-embeddings-inference/issues).
- Suggest or implement new features.
- Improve or extend the documentation.
- Review Pull Requests and assist other contributors.
- Spread the word: share TEI in blog posts or on social media, or give the repository a star ⭐.

Please take some time to review the relevant community guidelines before you start:

- [Code of conduct for Ascend open-source projects](https://atomgit.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)
- [Issue submission guide](https://atomgit.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)
- [Issue handling process description](https://atomgit.com/Ascend/community/blob/master/docs/contributor/issue-workflow-guidelines.md)
- [Ascend community developer testing contribution guide](https://atomgit.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)
- [Ascend open-source and third-party software repository creation and branch naming guidelines](https://atomgit.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)
- [Ascend open-source and third-party software management specifications](https://atomgit.com/Ascend/community/blob/master/docs/contributor/third-party-software-management-guide.md)
- [Community security design guideline](https://atomgit.com/Ascend/community/blob/master/docs/contributor/security-design-guideline.md)
- [Coding standards](https://atomgit.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- [Pull Request submission guide](https://atomgit.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)

## Development and Testing

1. **Preparing the development environment**

   Download the container image. The following commands use the TEI 26.0.0-910b-ubuntu22.04-py3.11 image as an example:
   ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11
    ```
   Run the container:
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
   Log in to the container:
   ```bash
   docker exec -it tei_container bash
   ```
2. **Forking the repository** <br>
On the AtomGit platform, click the Fork button in the upper-right corner of the repository to fork a copy of the source code into your personal repository.
3. **Cloning the code**<br>
In the container, clone your personal repository and start development.
   ```bash
   git clone https://atomgit.com/<your-username>/text-embeddings-inference.git
   ```
4. **Building the code**
   ```bash
   export PATH="/home/HwHiAiUser/.cargo/bin:$PATH"
   CARGO_BUILD_RUSTFLAGS="-C target-cpu=generic" bash -c "cargo install --path router -F python -F http --no-default-features"
   ```
5. **Verifying the functionality**
The following commands deploy the bge-large-zh-v1.5 model as an example:
   ```bash
    bash start.sh --model-id $MODEL_DIR/bge-large-zh-v1.5 --hostname 127.0.0.1 --port 8000
   ```
   - Access the service and verify that the service works as expected.
   ```bash
   curl 127.0.0.1:8080/v1/embeddings \
       -X POST \
       -d '{"input":["The capital of China is Beijing."]}' \
       -H 'Content-Type: application/json'
   ```
6. **Adding test cases**
   Add test cases in the `tests` directory to verify that the newly added functionality works as expected. If you need to run the full test suite, prepare the model weights used by all test cases in the directory specified by the `MODEL_DIR` environment variable.
   ```bash
      export MODEL_DIR=/home/data # Adjust the path based on the directory that stores your model weights
      bash run_presmoke.sh
   ```
7. **Running the pre-commit check**

   Before committing code locally, run the pre-commit check. For guidance, see the [pre-commit local running guide](https://atomgit.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md).

8. **Submitting a Pull Request**

   Submit the PR and wait for the code review.

If you have any questions or need help, feel free to ask in [Issues](https://atomgit.com/Ascend/text-embeddings-inference/issues) or contact us through other community channels.
