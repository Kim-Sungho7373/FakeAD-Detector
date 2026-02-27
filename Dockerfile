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

RUN playwright install-deps chromium
RUN playwright install chromium

# ==========================================
# 🌟 [핵심 수정] Hugging Face Spaces 유저 권한 설정
# ==========================================
# HF Spaces는 보안상 컨테이너 내부에서 root 실행을 차단합니다.
# 따라서 uid 1000번의 일반 유저를 생성하고 권한을 넘겨야 합니다.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 작업 디렉토리를 생성한 일반 유저의 폴더로 변경
WORKDIR $HOME/app

# [단계 5] 내 코드 복사 (소유권을 방금 만든 user로 지정)
COPY --chown=user:user . $HOME/app

# [핵심 수정] 파이썬 경로에 myapp 폴더 추가
# 변경된 작업 디렉토리($HOME/app)에 맞춰 경로를 업데이트합니다.
ENV PYTHONPATH="${PYTHONPATH}:$HOME/app/myapp"

# 외부 통신용 포트 개방
EXPOSE 7860

# [단계 6] 실행
CMD ["uvicorn", "myapp.app:app", "--host", "0.0.0.0", "--port", "7860"]