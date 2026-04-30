import subprocess
import sys
import time
import signal
import unittest
import requests
import numpy as np
import os


class TestBase(unittest.TestCase):
    """测试基类，提供公共的设置和清理方法"""
    model_id = None
    port = None
    env = None
    base_url = None
    process = None

    @classmethod
    def setUpClass(cls):
        model_dir=os.environ.get("MODEL_DIR", "/home/data")
        cls.process = _start_service(os.path.join(model_dir, cls.model_id), cls.port, env=cls.env)

    @classmethod
    def tearDownClass(cls):
        _stop_service(cls.process)


def _compute_cosine_distance(emb1, emb2):
    """计算两个嵌入向量的余弦距离"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def _wait_for_health(port, timeout=180, interval=5):
    """等待服务健康检查通过"""
    base_url = f"http://127.0.0.1:{port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def _start_service(model_id, port, env=None):
    """启动服务"""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    print(f"Starting service for model {model_id}")
    process = subprocess.Popen(
        [
            "/bin/bash",
            "start.sh",
            "--model-id",
            model_id,
            "--port",
            str(port),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=proc_env,
        shell=False,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    if not _wait_for_health(port):
        _stop_service(process)
        raise RuntimeError(
            f"Service for model '{model_id}' failed to start within the timeout"
        )
    return process


def _kill_child_processes(pid):
    """使用psutil递归终止子进程树"""
    try:
        import psutil
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            gone, alive = psutil.wait_procs(children, timeout=3)
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            parent.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass
    except ImportError:
        pass


def _kill_process_group(pid):
    """使用进程组方式终止子进程"""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


def _stop_service(process):
    """停止服务"""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # 如果terminate超时，使用进程组方式终止
            if hasattr(os, 'killpg'):
                _kill_process_group(process.pid)
                process.wait(timeout=10)
            else:
                process.kill()
                process.wait()
        finally:
            # 确保子进程被清理
            _kill_child_processes(process.pid)


def _make_test_class(model_id_, port_, env_=None, test_base_class=None, class_prefix="Test"):
    """创建测试类"""
    model_name = model_id_.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")
    class_name = f"{class_prefix}_{model_name}"

    base_class = test_base_class or TestBase
    
    class Cls(base_class):
        model_id = model_id_
        port = port_
        env = env_ or {}
        base_url = f"http://127.0.0.1:{port_}"

    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    return Cls