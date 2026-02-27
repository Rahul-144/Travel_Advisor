# Base image with Python
FROM python:3.10-slim

# avoid buffering so logs show up in real time
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system dependencies required by some python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libjq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy and install python requirements
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# copy project sources
COPY . ./

# by convention expose streamlit default port
EXPOSE 8501

# default command to start the Streamlit app
CMD ["streamlit", "run", "APP.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
