"""설치: 가상환경 만들기 -> 라이브러리 설치 -> 점검. 사용: python setup_env.py"""
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).parent
VENV_DIR = ROOT / ".venv"
SUPPORTED = ((3, 10), (3, 13))  # 3.12에서 확인함


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    if not SUPPORTED[0] <= sys.version_info[:2] <= SUPPORTED[1]:
        sys.exit(f"[실패] Python {sys.version_info[0]}.{sys.version_info[1]}은 지원하지 않습니다. 3.12를 설치하세요.")
    if not venv_python().exists():
        print("[1/3] 가상환경 만드는 중...", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    steps = (
        ("[2/3] 필요한 프로그램 설치 중... 몇 분 걸립니다.", ["-m", "pip", "install", "-r", "requirements.txt"]),
        ("[3/3] 설치 점검 중...", ["check.py"]),
    )
    for message, arguments in steps:
        print(message, flush=True)
        if subprocess.run([str(venv_python()), *arguments], cwd=ROOT).returncode != 0:
            sys.exit("[실패] 위에 나온 글자를 그대로 복사해서 보내주세요.")
    print("설치가 끝났습니다.")


if __name__ == "__main__":
    main()
