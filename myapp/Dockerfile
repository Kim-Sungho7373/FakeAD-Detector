# 파이썬 3.10 이미지 사용
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /code

# 시스템 패키지 설치 (packages.txt 읽기)
COPY packages.txt /code/packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 모든 파일 복사 (myapp 폴더가 있다면 해당 경로 주의)
COPY . .

# 포트 설정 (Hugging Face는 기본적으로 7860 포트를 사용합니다)
CMD ["uvicorn", "myapp.app:app", "--host", "0.0.0.0", "--port", "7860"]