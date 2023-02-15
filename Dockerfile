FROM python:3.10
RUN pip install --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD app.py .
EXPOSE 7860
CMD ["python", "app.py"]