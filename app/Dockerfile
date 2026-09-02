FROM python:3.11-slim 

WORKDIR /code 

COPY requirements-app.txt requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt 

COPY src/ src/
COPY app/ app/
COPY models/ models/ 
COPY data/processed/ data/processed/ 

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]