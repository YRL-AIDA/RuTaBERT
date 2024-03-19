FROM ubuntu
COPY . /app/rutabert/

WORKDIR /app/rutabert/

SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade -y &&\
    apt install -y python3 &&\
    apt install -y virtualenv && virtualenv venv &&\
    source venv/bin/activate && pip install -r requirements.txt

CMD source venv/bin/activate &&\
    python3 train.py 2> logs/error_train.log &&\
    python3 test.py 2> logs/error_test.log

