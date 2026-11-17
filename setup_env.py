from pathlib import Path
import shutil
import subprocess
import venv


def main():
    project = Path(__file__).resolve().parent
    env = project / ".venv"

    # 查找管理你当前 Python 的 uv。
    candidates = [
        shutil.which("uv"),
        str(Path.home() / ".local/bin/uv"),
        str(Path.home() / ".cargo/bin/uv"),
        "/opt/homebrew/bin/uv",
    ]
    uv = next(
        (path for path in candidates if path and Path(path).is_file()),
        None,
    )
    if uv is None:
        raise RuntimeError("未找到 uv，请把这条提示发给我。")

    # 不安装 pip，避开刚才失败的 ensurepip 步骤。
    print("正在准备虚拟环境……", flush=True)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(env)
    python = env / "bin" / "python"

    print("正在安装依赖……", flush=True)
    subprocess.run(
        [
            uv, "pip", "install",
            "--python", str(python),
            "numpy", "torch", "torchvision",
        ],
        check=True,
    )

    print("正在检查安装结果……", flush=True)
    subprocess.run(
        [
            str(python), "-c",
            "import numpy, torch, torchvision; print('依赖检查通过！')",
        ],
        check=True,
    )

    print("\n安装完成！请选择这个解释器：", flush=True)
    print(python)


if __name__ == "__main__":
    main()