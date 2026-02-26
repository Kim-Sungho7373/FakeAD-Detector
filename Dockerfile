FROM python:3.10

WORKDIR /code

# [단계 1] 시스템 패키지 설치
COPY packages.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    $(cat packages.txt) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# [단계 2] Pip 업그레이드 및 기본 도구 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# [단계 3] OpenCV 및 무거운 패키지 개별 설치 (충돌 방지)
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.kr/whl/linux/mkl/avx/stable.html
RUN pip install --no-cache-dir paddleocr

# [단계 4] 나머지 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [단계 5] 코드 복사 및 실행
COPY . .

CMD ["uvicorn", "myapp.app:app", "--host", "0.0.0.0", "--port", "7860"]