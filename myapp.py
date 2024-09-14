from fastapi import FastAPI, File, UploadFile, HTTPException
from llm.llm_config import llm
from botocore.exceptions import NoCredentialsError
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import boto3
import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_ID")
AWS_REGION = "eu-north-1" 
S3_BUCKET_NAME = "himanshu-rag"

myApp = FastAPI()

origins = [
    "localhost:3000",
    "http://localhost:3000",
]

myApp.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

@myApp.post("/llm/upload")
async def upload_file_to_s3(file: UploadFile = File(...)):
    try:
        # Upload the file to S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            file.filename,
            ExtraArgs={"ContentType": file.content_type}
        )
        return {"message": "File uploaded successfully", "filename": file.filename}
    
    except NoCredentialsError:
        raise HTTPException(status_code=403, detail="Credentials not available")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@myApp.get("/llm/search")
def search(query: str, fileName: str):
    rag_chain = llm().get_rag_chain(fileName)
    # Stream the response using StreamingResponse
    return EventSourceResponse(generate_ai_response(rag_chain, query))

def generate_ai_response(rag_chain, input_query):
    # Invoke the LLM chain
    try:
        res = rag_chain.invoke({"input": input_query})
        answer = res["answer"]
        print(answer)
        for chunk in answer.split():
            yield f"{chunk} "
        yield "\n"
    except Exception as e:
        print(e)
        yield f"Error: {e}"