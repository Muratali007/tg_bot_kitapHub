FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8181

ENV PYTHONUNBUFFERED=1

CMD ["python", "telegram.py"]