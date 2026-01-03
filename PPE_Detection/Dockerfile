FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY best.pt .
EXPOSE 8000

CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8000"]
