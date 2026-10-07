"""자체 시험. 실제 그림 없이 돈다. 사용: python test_workspace.py

맥에서 돌려도 윈도우에서 깨지기 쉬운 부분(한글·특수문자 경로, cp949 콘솔, 배치 파일 형식)을 함께 본다.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import check
import mosaic
import run

ROOT = Path(__file__).parent
SHAPE = (200, 200)
FAR = slice(0, 40)  # 가장 넓게 확장(36px)해도 닿지 않는 곳
CORE = slice(90, 110)


def detection(name: str, size: int, conf: float = 0.9) -> run.Detection:
    mask = np.zeros(SHAPE, bool)
    mask[CORE, CORE] = True
    return run.Detection(name, size, conf, mask)


ALL_SIZES = [detection("penis", size) for size in run.IMAGE_SIZES]
LEAK = [detection("penis", run.IMAGE_SIZES[0])]


def run_with_script(script: list[list[run.Detection]], folder: Path) -> tuple[str, str, Path, np.ndarray]:
    """run.detect가 script의 값을 순서대로 돌려주게 하고 process를 한 번 실행한다."""
    image = np.random.default_rng(1).integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
    source = folder / "원본 그림 (1).png"
    assert mosaic.write_png(source, image)
    remaining = list(script)
    originals = run.detect, run.OUT_DIR, run.HOLD_DIR
    run.detect = lambda *_: remaining.pop(0)
    run.OUT_DIR, run.HOLD_DIR = folder / "out", folder / "out" / "hold"
    try:
        status, detail = run.process(None, source)
    finally:
        run.detect, run.OUT_DIR, run.HOLD_DIR = originals
    assert not remaining, f"쓰이지 않은 검출 결과가 남음: {len(remaining)}"
    assert np.array_equal(mosaic.read_image(source), image), "원본이 바뀜"
    return status, detail, folder / "out", image


def test_state_machine() -> None:
    cases = (
        ("한 번에 통과", [ALL_SIZES, []], run.PASS, "원본 그림 (1)_model.png"),
        ("재검사에 걸린 뒤 통과", [ALL_SIZES, LEAK, ALL_SIZES, []], run.RETRIED, "원본 그림 (1)_model.png"),
        ("끝까지 걸림", [ALL_SIZES, LEAK] * len(run.ATTEMPTS), run.HOLD, "hold/원본 그림 (1)_model.png"),
        ("해상도 한 곳에서만 검출", [LEAK, []], run.HOLD, "hold/원본 그림 (1)_model.png"),
        ("검출 없음", [[]], run.NO_DETECTION, None),
    )
    for label, script, expected, saved_as in cases:
        with tempfile.TemporaryDirectory() as folder:
            status, detail, out_dir, image = run_with_script(script, Path(folder))
            assert status == expected, (label, status, detail)
            saved = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*.png")) if out_dir.exists() else []
            assert saved == ([saved_as] if saved_as else []), (label, saved)
            if saved_as:
                result = mosaic.read_image(out_dir / saved_as)
                assert result.shape == image.shape, label
                assert np.array_equal(result[FAR], image[FAR]), f"{label}: 가림 영역 밖이 바뀜"
                assert not np.array_equal(result[CORE, CORE], image[CORE, CORE]), f"{label}: 가려지지 않음"


def test_unreadable_file() -> None:
    with tempfile.TemporaryDirectory() as folder:
        broken = Path(folder) / "깨진 파일.png"
        broken.write_bytes(b"not an image")
        assert run.process(None, broken)[0] == run.ERROR
        assert run.process(None, Path(folder) / "없는 파일.png")[0] == run.ERROR


def test_expand_inputs() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name in ("가.png", "나.JPG", "메모.txt"):
            (root / name).write_bytes(b"x")
        assert [p.name for p in run.expand_inputs([folder])] == ["가.png", "나.JPG"]
        assert [p.name for p in run.expand_inputs([str(root / "*.png")])] == ["가.png"]
        assert run.expand_inputs([str(root / "가.png")]) == [root / "가.png"]


def test_summary_names_files_to_check() -> None:
    text = "\n".join(run.summary([("a.png", run.PASS), ("b.png", run.HOLD), ("c.png", run.NO_DETECTION)]))
    assert "통과 1" in text and "보류 1" in text and "검출 없음 1" in text
    assert "b.png" in text and "c.png" in text and "a.png" not in text


def test_cp949_console() -> None:
    """윈도우 한국어 콘솔(cp949)이 못 찍는 글자가 파일명에 있어도 끝까지 돌아야 한다. 실제 모델을 쓴다."""
    if not run.MODEL_PATH.exists():
        print("    (모델 파일이 없어 건너뜀)")
        return
    with tempfile.TemporaryDirectory() as folder:
        assert mosaic.write_png(Path(folder) / "흰 그림 😀.png", np.full((64, 64, 3), 255, np.uint8))
        done = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), folder],
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp949"},
        )
        output = done.stdout.decode("cp949")
        assert done.returncode == 0, done.stderr.decode(errors="replace")[-500:]
        assert "[검출 없음]" in output and "합계:" in output, output


def test_bat_files() -> None:
    """배치 파일은 윈도우에서 돌려보지 못하므로, 깨지기 쉬운 형식만이라도 고정한다."""
    for name in ("install.bat", "run.bat"):
        data = (ROOT / name).read_bytes()
        data.decode("ascii")  # 한글이 섞이면 콘솔 코드페이지에 따라 깨진다
        assert b"\n" not in data.replace(b"\r\n", b""), f"{name}: 줄 끝이 CRLF가 아님"
        assert b"(\r\n" not in data, f"{name}: 괄호 블록 안의 %*는 괄호가 든 파일명에서 깨진다"
        assert b'cd /d "%~dp0"' in data, f"{name}: 실행 위치를 폴더로 옮기지 않음"


def main() -> None:
    tests = (
        check.check_korean_path,
        check.check_mosaic,
        test_state_machine,
        test_unreadable_file,
        test_expand_inputs,
        test_summary_names_files_to_check,
        test_bat_files,
        test_cp949_console,
    )
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)}개 모두 통과")


if __name__ == "__main__":
    main()
