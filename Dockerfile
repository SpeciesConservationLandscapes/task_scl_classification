FROM scl3/task_base:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unixodbc-dev \
    freetds-dev \
    tdsodbc \
    git \
    less \
    nano \
    && apt-get clean -y

RUN pip install git+https://github.com/SpeciesConservationLandscapes/task_base.git \
    && pip install pandas==1.3.2 \
    && pip install pyodbc==4.0.32

WORKDIR /app
COPY $PWD/src .
COPY $PWD/odbcinst.ini /etc/odbcinst.ini
