FROM python:3.9-slim

# Expose port you want your app on
EXPOSE 8080

# # Optional - install git to fetch packages directly from github
# RUN apt-get update && apt-get install -y git

# Upgrade pip and install requirements
COPY requirements.txt requirements.txt
# RUN pip3 install -U pip3
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app code and set working directory
COPY SimpleModel.py SimpleModel.py
COPY app.py app.py

# Model
COPY model.joblib model.joblib

WORKDIR .

# Run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]