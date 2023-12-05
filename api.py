from fastapi import FastAPI, Body, Header 
from fastapi.responses import JSONResponse
import uvicorn
import gc
#local imports 
from embeddings import search , single_embedding_creation
from utility import check_credentials


#app initialize
app = FastAPI()

#response dict
response_dict={
    "Response":"",
    "Message":""
}

#face search endpoint
@app.post("/face_search")
async def face_search(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if check_credentials(userid,clientsecretkey):

            image_name = image_data.get('Image_Name', None)
            image_base=image_data.get('Image_Base64', None)
            
            output=search(image_base, search_type="face_search")

            response_dict["Response"]=output
            response_dict["Message"]="SUCCESS"
            return JSONResponse(output) 
        
        else:
            response_dict["Response"]="Try again"
            response_dict["Message"]="Failed. Invalid Header"
            gc.collect()
            return JSONResponse(response_dict)
    except Exception as e:
        
        response_dict["Response"]="Something went Wrong"
        response_dict["Message"]="Failed . Exception occured"
        gc.collect()
        return JSONResponse(response_dict)

@app.post("/create_embedding")
async def create_embedding(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if check_credentials(userid,clientsecretkey):

            img_name = image_data.get('Image_Name', None)
            img_base=image_data.get('Image_Base64', None)
            
            output=single_embedding_creation(img_bs64=img_base,image_name=img_name)

            # ES exception check
            if output=="Certain Error has occured. Check logs":
                response_dict["Response"]="Something went Wrong while indexing in Elasticsearch"
                response_dict["Message"]="Failed . Exception occured"
                gc.collect()
                return JSONResponse(response_dict)

            response_dict["Response"]=output
            response_dict["Message"]="SUCCESS"
            return JSONResponse(output) 
        
        else:
            response_dict["Response"]="Try again"
            response_dict["Message"]="Failed. Invalid Header"
            gc.collect()
            return JSONResponse(response_dict)
    
    except Exception as e:
        response_dict["Response"]="Something went Wrong"
        response_dict["Message"]="Failed . Exception occured"
        gc.collect()
        return JSONResponse(response_dict)

@app.post("/face_verify")
async def face_verify(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if check_credentials(userid,clientsecretkey):

            image_name = image_data.get('Image_Name', None)
            image_base=image_data.get('Image_Base64', None)
            
            output=search(image_base, search_type="face_verify")

            response_dict["Response"]=output
            response_dict["Message"]="SUCCESS"
            return JSONResponse(output) 
        
        else:
            response_dict["Response"]="Try again"
            response_dict["Message"]="Failed. Invalid Header"
            gc.collect()
            return JSONResponse(response_dict)
    except Exception as e:
        
        response_dict["Response"]="Something went Wrong"
        response_dict["Message"]="Failed . Exception occured"
        gc.collect()
        return JSONResponse(response_dict)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
