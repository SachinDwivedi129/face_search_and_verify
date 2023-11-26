# import necessary libraries
from elasticsearch import Elasticsearch
import configparser

# reading config file
config = configparser.ConfigParser()
config.read("config.ini")
 

# elastic connection 
def create_elasticsearch_connection():
    # Replace with your Elasticsearch server details
    es_host = config.get('elastic','host')
    es_port = int(config.get('elastic','port'))
    es_user = config.get('elastic','user')
    es_password = config.get('elastic','password')
    es_use_ssl = bool(config.get('elastic','use_ssl'))  # Set to True if your Elasticsearch cluster uses SSL
    es_ca_cert = config.get('elastic','ca_cert')  # Replace with the actual path

    # Create an Elasticsearch client with basic authentication and SSL
    es = Elasticsearch(
        [f'http://{es_host}:{es_port}'],
        basic_auth=(es_user, es_password),
        verify_certs=es_use_ssl,
        #ca_certs=es_ca_cert  # Provide the path to your certificate file
    )

    try:
        # Check the connection by sending a simple request
        info = es.info()
        print(f"Connected to Elasticsearch cluster: {info['cluster_name']}")

        return es

    except Exception as e:
        # log exception
        print(f"Failed to connect to Elasticsearch: {e}")


# index creation
def create_index(index_name:str , mapping: dict):
    es=create_elasticsearch_connection()
    """
    index_name="face_search"

    dense_vector_mapping = {
        "properties": {
            "embeddings": {
                "type": "dense_vector",
                "dims": 512  # The dimension of your face embeddings
            },
            "image_path": {
                "type": "keyword"
            }
        }
    }
    """
    return es.indices.create(index=index_name, body={"mappings": mapping})

# index data elasticsearxch index    
def index_data(data:dict , index_name:str):
    es=create_elasticsearch_connection()
    return es.index(index=index_name, body=data)
    

# elastic search with cosine similarity

def search_elastic(target_embeddings:list, similarity_threshold: float=1.5):
    try:
        es= create_elasticsearch_connection()
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {"query_vector": target_embeddings}
                    }
                }
            },
            "min_score":similarity_threshold
        }

        # Execute the search
        response = es.search(index="face_search", body=search_query, size=1000)
        
        # Retrieve the matching image path
        matching_image_path = [hit['_source']['image_path'] for hit in response['hits']['hits']]

        return matching_image_path
    except Exception as e:
        print(f"Exception as {e}")
        # log exception
        return "Error occured. Check logs"
