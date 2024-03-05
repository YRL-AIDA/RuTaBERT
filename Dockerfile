FROM ubuntu
COPY . /app/sti_team/cta/

# VOLUME ["/app/sti_team/cta/logs", "/app/sti_team/cta/checkpoints"]

WORKDIR /app/sti_team/cta/

SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade -y &&\
    apt install -y python3 &&\
    apt install -y virtualenv && virtualenv venv &&\
    source venv/bin/activate && pip install -r requirements.txt

CMD source venv/bin/activate; python3 train.py 2> logs/error.log
