# 예시 Dockerfile 핵심 내용
FROM python:3.11

# 시스템 패키지 설치 (packages.txt가 있다면 활용)
RUN apt-get update && apt-get install -y $(cat packages.txt)

WORKDIR /code
COPY . .

# 라이브러리 및 브라우저 설치
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

# 포트 설정 (허깅페이스 기본값)
EXPOSE 7860

# 실행 명령 (myapp 폴더 안의 app.py 실행)
CMD ["python", "myapp/app.py"]