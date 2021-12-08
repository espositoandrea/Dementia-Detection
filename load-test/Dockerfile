FROM python:3.7
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt 
COPY locustfile.py locustfile.py
EXPOSE 8089
CMD ["locust", "-f", "load-test/locustfile.py", "--host", "http://127.0.0.1:8080"]
