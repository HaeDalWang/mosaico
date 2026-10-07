"""마스크 영역에만 모자이크를 입힌다. 마스크 밖 픽셀은 원본 그대로 둔다."""
from pathlib import Path

import cv2
import numpy as np

BLOCK_PX = 18  # 모자이크 한 칸의 크기


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread는 윈도우에서 한글 경로를 못 읽는 경우가 있어서 바이트로 읽어 디코딩한다."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_png(path: Path, image: np.ndarray) -> bool:
    """cv2.imwrite 대신 인코딩한 바이트를 직접 쓴다. 이유는 read_image와 같다."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def apply_mosaic(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """격자 원점은 이미지 좌상단 (0,0), 칸 색은 칸 안 원본 픽셀의 평균(반올림)."""
    height, width = image.shape[:2]
    rows, cols = -(-height // BLOCK_PX), -(-width // BLOCK_PX)
    # 크기가 칸의 배수가 아니면 가장자리 칸이 잘린다. 잘린 칸은 실제 있는 픽셀만 평균낸다
    padded = np.full((rows * BLOCK_PX, cols * BLOCK_PX, image.shape[2]), np.nan)
    padded[:height, :width] = image
    means = np.nanmean(padded.reshape(rows, BLOCK_PX, cols, BLOCK_PX, -1), axis=(1, 3))
    blocky = np.rint(means).astype(image.dtype).repeat(BLOCK_PX, axis=0).repeat(BLOCK_PX, axis=1)
    result = image.copy()
    result[mask] = blocky[:height, :width][mask]
    return result
