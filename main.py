"""
main.py
-------
AI 분리수거 가이드 이미지 업로드 및 YOLO 판별 서버 (FastAPI)

애플리케이션 진입점(Entry Point):
1. FastAPI 애플리케이션 생성 및 수명 주기(Lifespan) 관리
2. YOLO 모델 사전 로딩
3. 라우터(routers/waste.py) 및 정적 파일 경로 마운트
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# YOLO 로딩 서비스 및 쓰레기 라우터 import
from yolo_service import load_model
from routers.waste import router as waste_router, MODEL_PATH, UPLOAD_DIR

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
)
logger = logging.getLogger("WasteServer")


# ==========================================
# 1. FastAPI 수명 주기(Lifespan) 관리
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 YOLO 모델을 메모리에 1회 미리 로드하고,
    서버 종료 시 자원을 정리합니다.
    """
    logger.info(">>> 서버 시작 중: YOLO 모델 로딩 준비...")

    if MODEL_PATH.exists():
        try:
            load_model(MODEL_PATH)
            logger.info(f">>> YOLO 모델 로드 성공: {MODEL_PATH}")
        except Exception as e:
            logger.error(f">>> YOLO 모델 로드 실패: {e}")
    else:
        logger.warning(f"[경고] 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        logger.warning("models/best.pt 파일을 배치해야 AI 판별 기능이 정상 동작합니다.")

    yield  # 서버 실행 중

    logger.info(">>> 서버가 종료되었습니다.")


# ==========================================
# 2. FastAPI 애플리케이션 생성
# ==========================================
app = FastAPI(
    title="Waste Image Upload & AI Detection Server",
    description="안드로이드 앱 연동 AI 분리수거 이미지 업로드 및 YOLO 판별 서버",
    version="2.0.0",
    lifespan=lifespan
)


# ==========================================
# 3. 라우터 등록 및 정적 파일 경로 마운트
# ==========================================
# 쓰레기 업로드, 판별, 박스 이미지 다운로드 API 등록 (/api/...)
app.include_router(waste_router)

# /uploads 경로로 정적 파일 제공 (앱에서 박스 쳐진 이미지를 다운로드/표시할 수 있음)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ==========================================
# 4. 서버 헬스체크 API
# ==========================================
@app.get("/health")
async def health_check():
    """
    서버 및 모델 로드 상태를 확인하는 헬스체크 엔드포인트
    """
    return {
        "status": "ok",
        "message": "Waste Detection Server is running",
        "modelLoaded": MODEL_PATH.exists()
    }
