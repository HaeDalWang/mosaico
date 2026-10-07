"""검출 모델로 가림 영역을 찾아 모자이크하고, 결과물을 다시 검사한다.

원본은 그대로 두고 out 폴더에 PNG로 저장한다. 재검사에서 걸리면 설정을 바꿔 자동으로 다시 가린다.
끝까지 걸리거나 검출이 불안정한 이미지는 out/hold 폴더에 따로 저장한다.

사용: python run.py [<이미지 또는 폴더> ...]
  아무것도 적지 않으면 input 폴더를 처리한다.
"""
import glob
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO

import mosaic

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models/v50/ntd11_anime_nsfw_segm_v5.pt"
MODEL_URL = "https://civitai.com/models/1313556"  # 모델은 저장소에 넣지 않는다. 받는 곳
INPUT_DIR = ROOT / "input"
OUT_DIR = ROOT / "out"
HOLD_DIR = OUT_DIR / "hold"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# ---- 조정할 값 ----
TARGET_CLASSES = frozenset({"penis", "testicles", "anus", "pussy"})
# 해상도 하나로는 부위를 통째로 놓치는 경우가 있어서 여러 해상도 결과를 합친다.
# 700 이상은 넣지 않는다. 이 모델은 큰 입력에서 부위를 놓치거나 엉뚱한 곳을 잡는다 (예시 1쌍에서 확인)
IMAGE_SIZES = (480, 544, 640)
# (검출 신뢰도 하한, 마스크 확장 px). 재검사에서 걸리면 다음 단계로 넘어가 더 넓게 가린다.
# 첫 단계의 18px는 모델 윤곽과 사람이 가린 윤곽의 차이에 맞춘 값 (예시 1쌍 기준)
ATTEMPTS = ((0.25, 18), (0.10, 27), (0.05, 36))
# 재검사: 결과물에서 이 신뢰도 이상으로 부위가 다시 보이면 불합격.
# 예시 1쌍에서 안 가린 부위는 0.9대, 제대로 가린 정답은 최대 0.48이 나와 그 사이로 잡았다
RECHECK_FAIL_CONF = 0.7
# -------------------

PASS, RETRIED, HOLD, NO_DETECTION, ERROR = "통과", "재시도 후 통과", "보류", "검출 없음", "오류"


class Detection(NamedTuple):
    name: str
    size: int
    conf: float
    mask: np.ndarray

    def __str__(self) -> str:
        return f"{self.name}@{self.size}:{self.conf:.2f}"


def detect(model: YOLO, image: np.ndarray, conf: float, classes: frozenset[str]) -> list[Detection]:
    shape = image.shape[:2]
    found = []
    for size in IMAGE_SIZES:
        result = model.predict(image, conf=conf, imgsz=size, retina_masks=True, verbose=False)[0]
        if result.masks is None:
            continue
        for seg, cls, score in zip(result.masks.data.cpu().numpy(), result.boxes.cls, result.boxes.conf):
            name = result.names[int(cls)]
            if name in classes:
                mask = cv2.resize(seg, shape[::-1], interpolation=cv2.INTER_NEAREST) > 0.5
                found.append(Detection(name, size, float(score), mask))
    return found


def union(detections: list[Detection], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    for detection in detections:
        mask = mask | detection.mask
    return mask


def expand(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def unstable_classes(detections: list[Detection]) -> list[str]:
    """여러 해상도 중 한 곳에서만 잡힌 부위. 모델이 애매하게 본 것이라 놓쳤을 가능성이 높다."""
    if len(IMAGE_SIZES) < 2:
        return []
    sizes_per_class: dict[str, set[int]] = {}
    for detection in detections:
        sizes_per_class.setdefault(detection.name, set()).add(detection.size)
    return sorted(name for name, sizes in sizes_per_class.items() if len(sizes) == 1)


def process(model: YOLO, image_path: Path) -> tuple[str, str]:
    """(상태, 설명)을 돌려준다."""
    image = mosaic.read_image(image_path)
    if image is None:
        return ERROR, "읽지 못함"
    shape = image.shape[:2]

    leak_mask = np.zeros(shape, bool)
    notes = []
    for attempt, (conf, expand_px) in enumerate(ATTEMPTS):
        detections = detect(model, image, conf, TARGET_CLASSES)
        if attempt == 0:
            if not detections:
                # 검출 0건을 "가릴 것 없음"으로 단정하지 않는다. 저장하지 않고 따로 보고한다
                return NO_DETECTION, "저장 안 함, 확인 필요"
            unstable = unstable_classes(detections)
            notes.append("검출 " + ", ".join(map(str, detections)))
        mask = expand(union(detections, shape) | leak_mask, expand_px)
        result = mosaic.apply_mosaic(image, mask)
        leaks = detect(model, result, RECHECK_FAIL_CONF, TARGET_CLASSES)
        if not leaks:
            break
        notes.append(f"{attempt + 1}차 재검사 불합격 " + ", ".join(map(str, leaks)))
        # 다시 보인 부위는 다음 시도에서 반드시 가려지도록 마스크에 더한다
        leak_mask = leak_mask | union(leaks, shape)

    if leaks:
        status = HOLD
        notes.append("재시도 후에도 부위가 다시 검출됨")
    elif unstable:
        status = HOLD
        notes.append(f"검출 불안정(해상도 한 곳에서만 잡힘): {', '.join(unstable)}")
    else:
        status = PASS if attempt == 0 else RETRIED

    out_dir = HOLD_DIR if status == HOLD else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_model.png"
    if not mosaic.write_png(out_path, result):
        return ERROR, f"저장 실패: {out_path}"
    saved = mosaic.read_image(out_path)
    if saved is None or saved.shape != image.shape or not np.array_equal(saved[~mask], image[~mask]):
        return ERROR, f"검증 실패: 크기 또는 가림 영역 밖 픽셀이 원본과 다름 ({out_path})"
    return status, f"{out_path.name} | 가림 {mask.sum()}px | " + " | ".join(notes)


def extract_model(archive: Path) -> Path | None:
    """zip 안의 .pt 파일을 제자리에 꺼낸다."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = sorted(name for name in bundle.namelist() if name.lower().endswith(".pt"))
            if not members:
                return None
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_PATH.write_bytes(bundle.read(members[0]))
    except (zipfile.BadZipFile, OSError):
        return None
    return MODEL_PATH


def find_model() -> Path | None:
    """모델 파일을 찾는다. 폴더명을 틀렸거나, 다른 이름이거나, zip째로 넣은 경우도 받아서 제자리에 둔다."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    models_dir = ROOT / "models"
    loose = sorted(models_dir.rglob("*.pt")) + sorted(ROOT.glob("*.pt"))
    if loose:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(loose[0], MODEL_PATH)
        print(f"모델 파일을 {loose[0]} 에서 찾아 제자리에 복사했습니다.", flush=True)
        return MODEL_PATH
    for archive in sorted(models_dir.rglob("*.zip")) + sorted(ROOT.glob("*.zip")):
        if extract_model(archive):
            print(f"모델 파일을 {archive} 에서 꺼내 제자리에 두었습니다.", flush=True)
            return MODEL_PATH
    return None


MODEL_MISSING = (
    "모델 파일이 없습니다. 공유받은 모델 파일(.pt 또는 zip)을 이 폴더의 models 폴더 안에 넣어주세요.\n"
    f"  직접 받으려면: {MODEL_URL} 의 v5.0"
)


def expand_inputs(args: list[str]) -> list[Path]:
    """폴더는 안의 이미지로, 와일드카드는 맞는 파일로 푼다. 윈도우 셸은 *.png 를 풀어주지 않는다."""
    paths = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            paths += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif any(ch in arg for ch in "*?["):
            paths += sorted(Path(p) for p in glob.glob(arg) if Path(p).suffix.lower() in IMAGE_SUFFIXES)
        else:
            paths.append(path)
    return paths


def summary(results: list[tuple[str, str]]) -> list[str]:
    """실행한 쪽(사람이든 에이전트든)이 다음에 무엇을 해야 하는지까지 출력에 담는다."""
    counts = Counter(status for _, status in results)
    lines = ["합계: " + ", ".join(f"{s} {counts[s]}" for s in (PASS, RETRIED, HOLD, NO_DETECTION, ERROR))]
    names = {s: [name for name, status in results if status == s] for s in (HOLD, NO_DETECTION, ERROR)}
    if counts[PASS] or counts[RETRIED]:
        lines.append(f"끝난 결과: {OUT_DIR}")
    if names[HOLD]:
        lines.append(f"직접 열어서 확인할 것 ({HOLD_DIR}): " + ", ".join(names[HOLD]))
    if names[NO_DETECTION]:
        lines.append("부위를 못 찾아 저장하지 않은 것 (가릴 것이 없는 그림인지 직접 확인): " + ", ".join(names[NO_DETECTION]))
    if names[ERROR]:
        lines.append("처리하지 못한 것: " + ", ".join(names[ERROR]))
    lines.append("참고: [통과]는 검출 모델의 재검사에 걸리지 않았다는 뜻이고, 노출이 전혀 없다는 보증은 아니다.")
    return lines


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # 콘솔이 못 찍는 글자가 파일명에 있어도 멈추지 않게
    args = sys.argv[1:]
    if not args:
        INPUT_DIR.mkdir(exist_ok=True)
        args = [str(INPUT_DIR)]
    paths = expand_inputs(args)
    if not paths:
        sys.exit(f"처리할 그림이 없습니다. {INPUT_DIR} 폴더에 그림을 넣거나, 그림 경로를 적어주세요.")
    if find_model() is None:
        sys.exit(MODEL_MISSING)
    model = YOLO(str(MODEL_PATH))
    results = []
    for path in paths:
        try:
            status, detail = process(model, path)
        except Exception as error:  # 한 장이 실패해도 나머지는 처리하고, 실패는 그대로 보고한다
            status, detail = ERROR, f"{type(error).__name__}: {error}"
        results.append((path.name, status))
        print(f"[{status}] {path.name}: {detail}", flush=True)
    print("\n" + "\n".join(summary(results)))


if __name__ == "__main__":
    main()
