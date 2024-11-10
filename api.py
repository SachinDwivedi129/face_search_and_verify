from fastapi import FastAPI, Body, Header , HTTPException
import base64
from fastapi.responses import JSONResponse
import uvicorn
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import gc
from embeddings import search
# from image_inap import Image_Moderation
# from PIL import Image

app = FastAPI()

response_dict={
    "Response":"",
    "Message":""
}
user="Event@kochi"
password="RandomGeneratedPassword@kochi"

@app.post("/face_verify")
async def face_verify_endpoint(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if userid is not None and clientsecretkey is not None:
            hdruserid = userid
            hdrclientsecretkey = clientsecretkey
        else:
            response_dict["Response"]="Try again"
            response_dict["Message"]="Failed. Invalid Header"
            gc.collect()
            return JSONResponse(response_dict)
        
        if hdruserid==user and hdrclientsecretkey==password:

            image_name = image_data.get('Image_Name', None)
            image_base=image_data.get('Image_Base64', None)
            
            output=search(image_base)

            response_dict["Response"]=output
            response_dict["Message"]="SUCCESS"
            return JSONResponse(output)
            
        
        else:
            response_dict["Response"]="Try again"
            response_dict["Message"]="Failed. Invalid Header"
            gc.collect()
            return JSONResponse(response_dict)
    except Exception as e:
        #raise HTTPException(status_code=500, detail=str(e))
        response_dict["Response"]="Something went Wrong"
        response_dict["Message"]="Failed . Exception occured"
        gc.collect()
        return JSONResponse(response_dict)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
