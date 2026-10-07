"""설치 점검. 그림 없이 환경만 확인한다. 사용: python check.py"""
import sys
import tempfile
from pathlib import Path


def check_korean_path() -> None:
    """한글 이름의 파일을 쓰고 다시 읽을 수 있는지. 윈도우에서 가장 흔히 깨지는 부분이다."""
    import numpy as np

    import mosaic

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "한글 이름 (1).png"
        assert mosaic.write_png(path, image), "한글 경로에 저장하지 못함"
        loaded = mosaic.read_image(path)
        assert loaded is not None and np.array_equal(loaded, image), "한글 경로에서 읽은 그림이 저장한 것과 다름"


def check_mosaic() -> None:
    """칸의 배수가 아닌 크기에서도 칸이 (0,0) 격자에 맞고, 마스크 밖은 그대로인지."""
    import numpy as np

    import mosaic

    block = mosaic.BLOCK_PX
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (2 * block + 4, 2 * block + 14, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], bool)
    mask[block:, block:] = True
    result = mosaic.apply_mosaic(image, mask)
    assert np.array_equal(result[~mask], image[~mask]), "마스크 밖 픽셀이 바뀜"
    full = image[block : 2 * block, block : 2 * block]
    assert (result[block : 2 * block, block : 2 * block] == np.rint(full.reshape(-1, 3).mean(0))).all()
    cut = image[2 * block :, 2 * block :]
    assert (result[2 * block :, 2 * block :] == np.rint(cut.reshape(-1, 3).mean(0))).all()


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError as error:
        sys.exit(f"[실패] 라이브러리가 없습니다: {error.name}. 설치를 다시 실행하세요.")
    import run

    try:
        check_korean_path()
        check_mosaic()
    except AssertionError as error:
        sys.exit(f"[실패] {error or '모자이크 계산이 예상과 다름'}")
    if not run.MODEL_PATH.exists():
        sys.exit(
            f"[실패] 모델 파일이 없습니다: {run.MODEL_PATH}\n"
            f"  {run.MODEL_URL} 에서 v5.0 을 받아 zip을 풀고, 안의 .pt 파일을 위 위치에 넣으세요."
        )
    model = YOLO(str(run.MODEL_PATH))
    missing = run.TARGET_CLASSES - set(model.names.values())
    if missing:
        sys.exit(f"[실패] 모델에 없는 가림 대상: {sorted(missing)}. 모델이 내는 이름: {sorted(model.names.values())}")
    blank = np.full((640, 480, 3), 255, np.uint8)
    if run.detect(model, blank, run.ATTEMPTS[0][0], run.TARGET_CLASSES):
        sys.exit("[실패] 아무것도 없는 흰 그림에서 부위가 검출됨. 모델 파일이 다른 것일 수 있습니다.")
    print("[준비 완료] 라이브러리, 모델, 한글 경로, 모자이크 계산이 모두 정상입니다.")


if __name__ == "__main__":
    main()
