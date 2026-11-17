"""
routers/waste.py
----------------
쓰레기 이미지 업로드, YOLO 판별, 박스 이미지 다운로드 등
'쓰레기 관련 API'들을 전담하는 라우터 모듈입니다.
"""

from pathlib import Path
from uuid import uuid4
import json
import logging
import mimetypes

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# YOLO 추론 전담 서비스 모듈 import
from yolo_service import predict_waste

# 로깅 설정
logger = logging.getLogger("WasteRouter")

# 라우터 생성 (prefix="/api" 설정으로 모든 경로는 /api/... 로 매핑됩니다)
router = APIRouter(prefix="/api", tags=["Waste Detection"])

# ==========================================
# 디렉터리 및 경로 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best.pt"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ==========================================
# 1. 이미지 업로드 및 쓰레기 판별 API
# ==========================================
@router.post("/upload")
async def upload_and_detect_waste(
    file: UploadFile = File(...)
):
    """
    안드로이드 앱에서 multipart/form-data 형태로 전송된 이미지를 수신하여:
    1. uploads/ 폴더에 원본 파일 저장
    2. 로드된 YOLO 모델로 쓰레기 판별 및 바운딩 박스가 그려진 이미지 생성
    3. 결과 JSON 및 박스 이미지 파일명을 반환
    """
    # 1. 파일 MIME 타입 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드할 수 있습니다."
        )

    # 2. 파일 확장자 검증
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다. ({', '.join(ALLOWED_EXTENSIONS)} 지원)"
        )

    # 3. 고유 파일명 생성 및 원본 저장
    saved_filename = f"waste_{uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / saved_filename
    file_size = 0

    try:
        with saved_path.open("wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                file_size += len(chunk)

    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        logger.error(f"이미지 저장 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail="이미지 저장 중 오류가 발생했습니다."
        ) from e
    finally:
        await file.close()

    # 4. YOLO 모델을 통한 쓰레기 판별 및 박스 이미지 생성
    prediction_result = None
    annotated_filename = None
    annotated_image_url = None

    try:
        if MODEL_PATH.exists():
            # YOLO 추론 실행 및 박스 그려진 이미지 저장 (save_annotated=True)
            prediction_result = predict_waste(saved_path, conf_threshold=0.25, save_annotated=True)
            annotated_filename = prediction_result.get("annotated_filename")

            if annotated_filename:
                annotated_image_url = f"/api/annotated/{annotated_filename}"

            # 콘솔 터미널에 판별 결과 출력
            if prediction_result.get("detected"):
                top = prediction_result["top_prediction"]
                logger.info("=" * 50)
                logger.info(f"🎯 [AI 쓰레기 판별 완료]")
                logger.info(f"   - 원본 파일명 : {saved_filename}")
                logger.info(f"   - 박스 파일명 : {annotated_filename}")
                logger.info(f"   - 최우선 분류 : {top['class_name']} ({top['confidence'] * 100:.1f}%)")
                logger.info(f"   - 감지 개수   : {prediction_result['total_count']}개")
                logger.info("=" * 50)
            else:
                logger.info(f"🎯 [AI 판별 결과] 감지된 쓰레기가 없습니다. (파일명: {saved_filename})")

            # 판별 결과를 uploads 폴더에 .json 파일로도 저장
            json_path = saved_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(prediction_result, f, ensure_ascii=False, indent=2)

        else:
            logger.warning("모델 파일(models/best.pt)이 없어 AI 판별을 건너뜁니다.")
    except Exception as e:
        logger.error(f"YOLO 판별 중 오류 발생: {e}")
        prediction_result = {
            "error": "AI 판별 도중 오류가 발생했습니다.",
            "details": str(e)
        }

    # 5. 최종 응답 JSON 구성
    return {
        "success": True,
        "filename": saved_filename,
        "annotatedFilename": annotated_filename,
        "annotatedImageUrl": annotated_image_url,
        "originalFilename": file.filename,
        "contentType": file.content_type,
        "sizeBytes": file_size,
        "prediction": prediction_result,
        "message": "사진 업로드 및 쓰레기 판별이 완료되었습니다."
    }


# ==========================================
# 2. 기존 업로드된 이미지 재판별 API
# ==========================================
@router.post("/predict/{filename}")
async def predict_existing_image(
    filename: str,
    conf: float = 0.25
):
    """
    이미 업로드되어 있는 이미지 파일명을 받아 다시 YOLO 판별 및 박스 이미지를 생성합니다.
    """
    target_path = UPLOAD_DIR / filename
    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail="요청한 이미지 파일을 찾을 수 없습니다."
        )

    try:
        prediction_result = predict_waste(target_path, conf_threshold=conf, save_annotated=True)
        annotated_filename = prediction_result.get("annotated_filename")
        annotated_image_url = f"/api/annotated/{annotated_filename}" if annotated_filename else None

        return {
            "success": True,
            "filename": filename,
            "annotatedFilename": annotated_filename,
            "annotatedImageUrl": annotated_image_url,
            "prediction": prediction_result
        }
    except Exception as e:
        logger.error(f"재판별 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"판별 중 오류 발생: {str(e)}"
        )


# ==========================================
# 3. 바운딩 박스 이미지 다운로드/조회 API
# ==========================================
@router.get("/annotated/{filename}")
async def get_annotated_image(filename: str):
    """
    안드로이드 앱에서 바운딩 박스가 그려진 이미지를 다운로드하거나
    ImageView(Glide/Coil)에 표시할 때 사용하는 파일 스트리밍 엔드포인트
    """
    # 1. 파일명 경로 조작(..) 방지 검증
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="잘못된 파일명입니다."
        )

    # 2. 허용된 이미지 확장자 확인
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 이미지 형식입니다."
        )

    # 3. 실제 이미지 저장 경로 계산
    image_path = (UPLOAD_DIR / filename).resolve()

    # 4. uploads 디렉터리 밖의 파일 접근 차단 (Directory Traversal 방지)
    if image_path.parent != UPLOAD_DIR.resolve():
        raise HTTPException(
            status_code=400,
            detail="잘못된 파일 경로입니다."
        )

    # 5. 이미지 파일 존재 여부 확인
    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="분석된 이미지를 찾을 수 없습니다."
        )

    # 6. 이미지 MIME 타입 확인 (image/jpeg 등)
    media_type = (
        mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )

    # 7. 이미지 파일 바이너리를 HTTP 응답으로 반환
    return FileResponse(
        path=image_path,
        media_type=media_type,
        filename=filename
    )
