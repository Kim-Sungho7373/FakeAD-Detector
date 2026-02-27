FROM python:3.10

# 패키지 설치는 root 권한으로 진행하기 위해 작업 디렉토리 설정
WORKDIR /code

# [단계 1] 시스템 패키지 설치
COPY packages.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    $(cat packages.txt) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# [단계 2] 기본 도구 업데이트
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# [단계 3] AI 이미지 처리 관련 무거운 라이브러리 먼저 설치 (OCR 용도)
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir paddlepaddle==3.0.0
RUN pip install --no-cache-dir "paddleocr>=2.7.0"

# [단계 4] 나머지 일반 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🚨 [수정 포인트 1] 루트(root) 권한일 때 OS 시스템 의존성만 설치
RUN playwright install-deps chromium

# ==========================================
# 🌟 [핵심 수정] Hugging Face Spaces 유저 권한 설정
# ==========================================
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 🚨 [수정 포인트 2] 유저(user)로 권한이 바뀐 직후에 브라우저 실제 파일 다운로드!
# 이렇게 해야 /home/user/.cache/ms-playwright/ 위치에 정확히 설치됩니다.
RUN playwright install chromium

# 작업 디렉토리를 생성한 일반 유저의 폴더로 변경
WORKDIR $HOME/app

# [단계 5] 내 코드 복사 (소유권을 방금 만든 user로 지정)
COPY --chown=user:user . $HOME/app

# [단계 6] 실행
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]