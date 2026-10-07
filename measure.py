"""사람이 가린 전/후 쌍으로 run.py의 결과를 숫자로 평가한다. 그림을 화면에 띄우지 않는다.

사용: python measure.py <원본 폴더> <사람이 가린 폴더>
  두 폴더의 이미지를 파일명 순서대로 짝짓는다. 장수가 같아야 한다.
  값(run.py의 ATTEMPTS, IMAGE_SIZES 등)을 바꾼 뒤에는 이 평가를 다시 돌려 전체 숫자로 비교한다.
"""
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import mosaic
import run

# PNG 쌍 기준. 올리면 단색 면의 가림(차이가 작은 픽셀)을 놓쳐 정답 영역이 작게 잡힌다. JPEG 쌍이면 올릴 것
DIFF_THRESHOLD = 0


def changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """전/후 픽셀 차이 = 실제로 가려진 영역."""
    diff = cv2.absdiff(before, after).max(axis=2) > DIFF_THRESHOLD
    # 모자이크 칸 안에서 우연히 원본과 같은 픽셀이 구멍으로 남으므로 한 칸 크기로 메운다
    kernel = np.ones((mosaic.BLOCK_PX, mosaic.BLOCK_PX), np.uint8)
    return cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in run.IMAGE_SUFFIXES)


def evaluate_pair(model: YOLO, before_path: Path, after_path: Path) -> tuple[str, float, float] | str:
    """(상태, 덮은 비율, 초과분) 또는 평가하지 못한 이유."""
    before, after = mosaic.read_image(before_path), mosaic.read_image(after_path)
    if before is None or after is None:
        return "이미지를 읽지 못함"
    if before.shape != after.shape:
        return f"크기 불일치 {before.shape[:2]} vs {after.shape[:2]}: 짝이 맞는지 확인"
    truth = changed_mask(before, after)
    if not truth.any():
        return "전/후가 같음: 사람이 가린 영역이 없음"
    status, detail = run.process(model, before_path)
    if status in (run.NO_DETECTION, run.ERROR):
        return f"[{status}] {detail}"
    out_dir = run.HOLD_DIR if status == run.HOLD else run.OUT_DIR
    result = mosaic.read_image(out_dir / f"{before_path.stem}_model.png")
    mine = changed_mask(before, result)
    return status, (truth & mine).sum() / truth.sum(), (mine & ~truth).sum() / max(mine.sum(), 1)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    befores, afters = images_in(Path(sys.argv[1])), images_in(Path(sys.argv[2]))
    if not befores or len(befores) != len(afters):
        sys.exit(f"원본 {len(befores)}장, 가린 것 {len(afters)}장: 장수가 같아야 합니다.")
    model = YOLO(str(run.MODEL_PATH))

    scores = []
    for before_path, after_path in zip(befores, afters):
        outcome = evaluate_pair(model, before_path, after_path)
        if isinstance(outcome, str):
            print(f"{before_path.name}: 평가 못 함: {outcome}")
            continue
        status, covered, excess = outcome
        scores.append((covered, excess))
        print(f"{before_path.name}: [{status}] 사람 영역 중 덮은 비율 {covered:.1%}, 초과분 {excess:.1%}")

    if not scores:
        sys.exit("평가된 쌍이 없습니다.")
    covered, excess = zip(*scores)
    print(
        f"\n{len(scores)}/{len(befores)}쌍 평가 | 덮은 비율 평균 {statistics.mean(covered):.1%}, "
        f"최저 {min(covered):.1%} | 초과분 평균 {statistics.mean(excess):.1%}, 최고 {max(excess):.1%}"
    )


if __name__ == "__main__":
    main()
