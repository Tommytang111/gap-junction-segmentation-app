FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
