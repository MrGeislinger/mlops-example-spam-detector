# Docker

> Setup using Docker 

## Local

* Build image: `docker build . -t spam-detector-image`
* Container: `docker run -p 8080:8080 --name spam-detector-container spam-detector-image`
* Go to URL (like `http://0.0.0.0:8080/`)


## GCP

> Make sure you have a GCP project setup already

### Cloud Run

* (Move files)
* (not needed?) `docker build . -t spam-detector-image`
* `docker tag spam-detector-image gcr.io/{PROJECT_ID}/spam-detector:vX`
* (Best if Docker image pushed to Google Cloud Registry - GCR)
    - (Allow for connection of Docker to GCR)
    - `docker push gcr.io/{PROJECT_ID}/spam-detector:v2` (note `gcr.io` is based on region)
        - https://support.terra.bio/hc/en-us/articles/360035638032-Publish-a-Docker-container-image-to-Google-Container-Registry-GCR-
        - https://cloud.google.com/container-registry/docs/advanced-authentication#console_2
* Create service in Cloud Run
* Use Docker image created for run
