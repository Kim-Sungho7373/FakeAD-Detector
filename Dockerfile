FROM python:3.10

WORKDIR /code

# 시스템 패키지 설치
COPY packages.txt /code/packages.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    $(cat packages.txt) \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY . .

# 실행 경로 (myapp 폴더 안에 app.py가 있는 경우)
CMD ["uvicorn", "myapp.app:app", "--host", "0.0.0.0", "--port", "7860"]