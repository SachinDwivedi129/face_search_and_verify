# import necessary libraries
from insightface.app import FaceAnalysis
import cv2
import os 
import configparser
import base64
import numpy as np
import time
# imports from local file
from elastic import index_data ,search_elastic

# reading config file
config = configparser.ConfigParser()
config.read("config.ini")

model_start=time.time()
# defining the insightface model
model = FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(480, 480))

print(f"model time {time.time()- model_start}")

# image preprocessing
def image_pre_process(image_path:str or bool=False , bs64:str or bool=False):
    if bs64==False:
        original_image = cv2.imread(image_path)
        # print(original_image.shape)
        # #original_image= cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        decoded_image = base64.b64decode(bs64)
        np_arr = np.frombuffer(decoded_image, dtype=np.uint8)
        original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    denoised_image = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 15)
    return denoised_image

# image directory example
image_dir=config.get("image","image_dir")

# create bulk embeddings
def create_bulk_embeddings(image_dir:str):
    
    list_images=os.listdir(image_dir)

    inside_start=time.time()
    i=1
    for image in list_images:
        image_path=os.path.join(image_dir,image)
        print(f"image_path=========> {image_path}")
        
        denoised_image=image_pre_process(image_path, bs64=False)
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
            
    print(f"inside_final {time.time() -inside_start}")
    return "Data indexed successfully"
                
def single_embedding_creation(img_bs64:str, image_name:str):
    denoised_image= image_pre_process(img_bs64,image_path=False)

    faces =model.get(denoised_image)

    for face in range(len(faces)):

        data={
            "embeddings":faces[face]["embedding"].tolist(),
            "image_name":image_name   
        }
        
        try:
            print(index_data(data,index_name="face_verification"))
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

def search(bs64:str):
    
    # doing pre processing
    denoised_image= image_pre_process(bs64=bs64, image_path=False)
    # getting image embeddings
    target_embeddings =model.get(denoised_image)
    
    # converting to list to search in elastic
    embeddings=target_embeddings[0]["embedding"].tolist()

    # threshold 
    threshold = float(config.get("image","threshold"))

    output=search_elastic(target_embeddings=embeddings, similarity_threshold=threshold)

    return output


if __name__=="__main__":
    overall_start=time.time()
    result=create_bulk_embeddings("../Celebrity/multi/")
    print(f"Overall time {time.time() -overall_start}")
    if result=="Certain Error has occured. Check logs":
        print("Check nohup.out . Something went wrong")
    else:
        print(result)
