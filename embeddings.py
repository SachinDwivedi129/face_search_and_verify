# import necessary libraries
from insightface.app import FaceAnalysis
import cv2
import os 
import configparser
# imports from local file
from elastic import index_data ,search_elastic

# reading config file
config = configparser.ConfigParser()
config.read("config.ini")

# defining the insightface model
model = FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(480, 480))



# image preprocessing
def image_pre_process(image_path:str):
    original_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    denoised_image = cv2.fastNlMeansDenoisingColored(rgb_image, None, 10, 10, 7, 15)
    return denoised_image

# image directory example
image_dir=config.get("image","image_dir")

# create bulk embeddings
def create_bulk_embeddings(image_dir:str):
    
    list_images=os.listdir(image_dir)

    for image in list_images:
        image_path=os.path.join(image_dir,image)
        print(f"image_path=========> {image_path}")
        
        denoised_image=image_pre_process(image_path)
        faces =model.get(denoised_image)

        for face in range(len(faces)):

            data={
                "embeddings":faces[face]["embedding"].tolist(),
                "image_path":image_path   
            }
            
            try:
                print(index_data(data,"face_search"))
                print(f"Number=====>{i}")
                i+=1
                
            except Exception as e:
                print(f"Number=====>{i}")
                i+=1
                print("Exception ======>",e)
                # will log exception 
                return "Certain Error has occured. Check logs"
            

        return "Data indexed successfully"
                

# search functionality

def search(image_path:str):
    # doing pre processing
    denoised_image= image_pre_process(image_path)

    # getting image embeddings
    target_embeddings =model.get(denoised_image)
    
    # converting to list to search in elastic
    embeddings=target_embeddings[0]["embedding"].tolist()

    # threshold 
    threshold = float(config.get("image","threshold"))

    output=search_elastic(target_embeddings=embeddings, similarity_threshold=threshold)

    return output
