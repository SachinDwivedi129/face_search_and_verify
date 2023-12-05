from fastapi import FastAPI, Body, Header 
from fastapi.responses import JSONResponse
import uvicorn
import gc
from embeddings import search
#local import 
from utility import check_credentials


#app initialize
app = FastAPI()

#response dict
response_dict={
    "Response":"",
    "Message":""
}


@app.post("/face_search")
async def extract_text_from_pdf(image_data: dict = Body(...), userid: str = Header(None), clientsecretkey: str = Header(None)):
    try:
        if check_credentials(userid,clientsecretkey):

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
        
        response_dict["Response"]="Something went Wrong"
        response_dict["Message"]="Failed . Exception occured"
        gc.collect()
        return JSONResponse(response_dict)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
