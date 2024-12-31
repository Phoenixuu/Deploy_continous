# Chọn base image
FROM python:3.8-slim

# Cài đặt các thư viện cần thiết
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Copy mã nguồn vào container
COPY . /app

# Mở port cho API (nếu triển khai API)
EXPOSE 5000

# Chạy mô hình hoặc API
CMD ["python", "randomForest.py"]

