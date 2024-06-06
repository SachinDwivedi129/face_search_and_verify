# face_search using elasticsearch


This Python application demonstrates how to store face embeddings in Elasticsearch as dense vectors and retrieve images based on these embeddings.

## Prerequisites

1. Python Dependencies: Install the required Python packages by running:

    ```bash
    pip install -r requirements.txt
    ```


2. Elasticsearch Setup:

   - Follow the instructions to set up Elasticsearch using Docker. If you don't have Docker installed, download and install it from [Docker's official website](https://docs.docker.com/get-docker/).

   - Run the following command to start an Elasticsearch instance:

     ```bash
     docker run --name es01 -p 9200:9200 -v elasticsearch_data:/usr/share/elasticsearch/data -d --net elastic docker.elastic.co/elasticsearch/elasticsearch:8.11.0
     ```

   - For detailed instructions, refer to the [Elasticsearch Docker Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html).

## Next step 

  - Coming soon..
