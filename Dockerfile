FROM python:3.10

WORKDIR /code

# [단계 1] 시스템 패키지 설치
COPY packages.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    $(cat packages.txt) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# [단계 2] 기본 도구 업데이트
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# [단계 3] AI 이미지 처리 관련 무거운 라이브러리 먼저 설치
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir paddlepaddle==3.0.0
RUN pip install --no-cache-dir paddleocr>=2.7.0

# [단계 4] 나머지 일반 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [단계 5] 내 코드 복사
COPY . .

# [핵심 수정] 파이썬 경로에 myapp 폴더 추가
# 이를 통해 app.py에서 'from step0_ingestion'을 바로 호출할 수 있게 됩니다.
ENV PYTHONPATH="${PYTHONPATH}:/code/myapp"

# [단계 6] 실행
CMD ["uvicorn", "myapp.app:app", "--host", "0.0.0.0", "--port", "7860"]