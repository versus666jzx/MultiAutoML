FROM python:3.8-slim
RUN /usr/local/bin/python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    rm requirements.txt && \
    mkdir /opt/multiautoml
COPY src/* /opt/multiautoml
WORKDIR /opt/multiautoml
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "80"]