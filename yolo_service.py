
"""
yolo_service.py
---------------
YOLO AI 모델을 로드하고 이미지 추론(Prediction),
NMS 후처리 및 Annotated 이미지 저장을 담당하는 서비스 모듈입니다.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# YOLO 모델 객체를 담아둘 전역 변수
_yolo_model = None


# ==========================================
# YOLO 추론 테스트 설정
# ==========================================

DEFAULT_CONF_THRESHOLD = 0.70
DEFAULT_IOU_THRESHOLD = 0.5

# True: 서로 다른 클래스 사이의 중복 박스도 억제
# False: 같은 클래스 내부에서만 중복 박스 억제
DEFAULT_AGNOSTIC_NMS = True


def load_model(model_path: Path):
    """
    YOLO 모델(.pt)을 메모리에 로드합니다.
    서버 시작 시 한 번만 호출하는 것을 권장합니다.
    """
    global _yolo_model

    try:
        from ultralytics import YOLO

    except ImportError as e:
        logger.error(
            "ultralytics 패키지가 설치되지 않았습니다. "
            "pip install ultralytics 를 실행해주세요."
        )
        raise e

    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}"
        )

    logger.info(f"YOLO 모델 로딩 시작: {model_path}")

    _yolo_model = YOLO(str(model_path))

    logger.info("YOLO 모델 로딩 완료!")

    return _yolo_model


def get_model():
    """
    현재 로드된 YOLO 모델 인스턴스를 반환합니다.
    """
    global _yolo_model

    return _yolo_model


def predict_waste(
    image_path: Path,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    save_annotated: bool = True,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    agnostic_nms: bool = DEFAULT_AGNOSTIC_NMS
) -> Dict[str, Any]:
    """
    저장된 폐기물 이미지를 YOLO 모델로 추론합니다.

    NMS 설정을 적용하여 중복 탐지를 억제하고,
    분석 결과와 Annotated 이미지를 반환합니다.

    Args:
        image_path:
            분석할 이미지 경로

        conf_threshold:
            최소 Confidence 기준

        save_annotated:
            Annotated 이미지 저장 여부

        iou_threshold:
            NMS의 IoU 기준

        agnostic_nms:
            True이면 서로 다른 클래스 간 중복도 억제

    Returns:
        detected:
            객체 탐지 여부

        total_count:
            NMS 적용 후 최종 탐지 박스 개수

        top_prediction:
            신뢰도가 가장 높은 탐지 결과

        predictions:
            모든 탐지 결과 목록

        annotated_filename:
            Annotated 이미지 파일명
    """

    # ==========================================
    # 1. 모델 및 이미지 확인
    # ==========================================

    model = get_model()

    if model is None:
        raise RuntimeError(
            "YOLO 모델이 아직 로드되지 않았습니다. "
            "load_model()을 먼저 호출해주세요."
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"추론할 이미지 파일을 찾을 수 없습니다: {image_path}"
        )

    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold는 0~1 사이여야 합니다.")

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold는 0~1 사이여야 합니다.")

    logger.info(
        "YOLO 추론 시작 | "
        f"conf={conf_threshold}, "
        f"iou={iou_threshold}, "
        f"agnostic_nms={agnostic_nms}"
    )

    # ==========================================
    # 2. YOLO 모델 추론 및 NMS 후처리
    # ==========================================

    results = model.predict(
        source=str(image_path),

        # 신뢰도 임계값
        conf=conf_threshold,

        # NMS IoU 임계값
        iou=iou_threshold,

        # 클래스 간 중복 억제 여부
        agnostic_nms=agnostic_nms,

        verbose=False
    )

    detected_list: List[Dict[str, Any]] = []

    annotated_filename: Optional[str] = None

    # ==========================================
    # 3. 추론 결과 파싱
    # ==========================================

    for result in results:

        boxes = result.boxes

        names = result.names

        for box in boxes:

            # 클래스 ID
            cls_id = int(box.cls[0].item())

            # AI 신뢰도
            confidence = float(box.conf[0].item())

            # 클래스 이름
            class_name = names.get(
                cls_id,
                f"class_{cls_id}"
            )

            # 바운딩 박스 좌표
            xyxy = [
                round(float(coord), 2)
                for coord in box.xyxy[0].tolist()
            ]

            detected_list.append({

                "class_id": cls_id,

                "class_name": class_name,

                "confidence": round(confidence, 4),

                "box": {
                    "xmin": xyxy[0],
                    "ymin": xyxy[1],
                    "xmax": xyxy[2],
                    "ymax": xyxy[3]
                }

            })

        # ==========================================
        # 4. Annotated 이미지 저장
        # ==========================================

        if save_annotated:

            annotated_filename = (
                f"{image_path.stem}_annotated{image_path.suffix}"
            )

            annotated_save_path = (
                image_path.parent / annotated_filename
            )

            result.save(
                filename=str(annotated_save_path)
            )

            logger.info(
                f"Annotated 이미지 저장 완료: {annotated_filename}"
            )

    # ==========================================
    # 5. 신뢰도 기준 정렬
    # ==========================================

    detected_list.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    # ==========================================
    # 6. 최우선 탐지 결과 선정
    # ==========================================

    top_prediction = (
        detected_list[0]
        if detected_list
        else None
    )

    # ==========================================
    # 7. 분석 결과 로그
    # ==========================================

    logger.info(
        f"최종 탐지 박스 개수: {len(detected_list)}"
    )

    for prediction in detected_list:

        logger.info(
            f"탐지 결과: {prediction['class_name']} | "
            f"Confidence: {prediction['confidence']}"
        )

    # ==========================================
    # 8. 결과 반환
    # ==========================================

    return {

        "detected": len(detected_list) > 0,

        "total_count": len(detected_list),

        "top_prediction": top_prediction,

        "predictions": detected_list,

        "annotated_filename": annotated_filename

    }
