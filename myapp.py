from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from llm.llm_config import llm
from llm.financial_processor import (
    extract_tables_from_pdf,
    extract_images_from_pdf,
    analyze_image_with_gemini,
    analyze_financial_document,
    analyze_table
)
from llm.response_formatter import format_response, format_for_chat
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Request models
class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None

class ChatRequest(BaseModel):
    id: Optional[str] = None
    messages: List[Message]
    trigger: Optional[str] = None
    fileName: Optional[str] = None  # Single file (backward compatible)
    fileNames: Optional[List[str]] = None  # Multiple files
    analysisType: Optional[str] = None  # "financial", "general", etc.
    responseFormat: Optional[str] = "text"  # "text", "json", "markdown"

class FinancialAnalysisRequest(BaseModel):
    fileNames: List[str]
    query: Optional[str] = None
    extractTables: bool = True
    analyzeCharts: bool = True
    responseFormat: str = "detailed"  # "summary", "detailed", "markdown", "structured"

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "himanshu-rag")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Local fallback directory
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize GCS
gcs_client = None
bucket = None
gcs_error = None
gcs_available = False

def initialize_gcs():
    """Initialize GCS client and bucket"""
    global gcs_client, bucket, gcs_error, gcs_available
    
    try:
        # Check if service account key file is provided
        if GOOGLE_APPLICATION_CREDENTIALS:
            if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
                raise FileNotFoundError(f"Service account key file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        
        # Initialize client
        if GCP_PROJECT_ID:
            gcs_client = storage.Client(project=GCP_PROJECT_ID)
            print(f"🔧 Initializing GCS with project: {GCP_PROJECT_ID}")
        else:
            gcs_client = storage.Client()
            print(f"🔧 Initializing GCS with default credentials")
        
        # Get bucket reference
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        
        # Test connection by checking if bucket exists
        if not bucket.exists():
            raise Exception(f"Bucket '{GCS_BUCKET_NAME}' does not exist. Please create it in GCP Console.")
        
        # Test write permissions by checking bucket metadata
        bucket.reload()
        
        gcs_available = True
        print(f"✅ GCS connected successfully to bucket: {GCS_BUCKET_NAME}")
        return True
        
    except FileNotFoundError as e:
        gcs_error = str(e)
        print(f"❌ GCS initialization failed: {e}")
        return False
    except GoogleCloudError as e:
        gcs_error = str(e)
        error_code = getattr(e, 'code', None)
        if error_code == 403:
            print(f"❌ GCS access denied (403). Check:")
            print(f"   1. Billing is enabled for your GCP project")
            print(f"   2. Service account has Storage Admin role")
            print(f"   3. Bucket permissions are correct")
        else:
            print(f"❌ GCS error ({error_code}): {e}")
        return False
    except Exception as e:
        gcs_error = str(e)
        print(f"⚠️ GCS not available: {e}")
        print(f"   Using local storage fallback")
        return False

# Initialize GCS on startup
initialize_gcs()

myApp = FastAPI()

origins = [
    "localhost:3000",
    "http://localhost:3000",
    "http://localhost:8000",
    "*",  # Allow all for development
]

myApp.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@myApp.get("/health")
def health():
    return {"status": "ok"}

@myApp.get("/llm/gcs/status")
def gcs_status():
    """Check GCS connection status and configuration"""
    status = {
        "available": gcs_available,
        "bucket_name": GCS_BUCKET_NAME,
        "project_id": GCP_PROJECT_ID if GCP_PROJECT_ID else "Using default credentials",
        "error": gcs_error if not gcs_available else None
    }
    
    if gcs_available and bucket:
        try:
            # Test bucket access
            bucket.reload()
            status["bucket_exists"] = True
            status["bucket_location"] = bucket.location
        except Exception as e:
            status["bucket_exists"] = False
            status["error"] = str(e)
    
    return status

@myApp.post("/llm/gcs/reinitialize")
def reinitialize_gcs():
    """Reinitialize GCS connection (useful after fixing billing/permissions)"""
    global gcs_client, bucket, gcs_error, gcs_available
    
    result = initialize_gcs()
    
    return {
        "success": result,
        "available": gcs_available,
        "bucket_name": GCS_BUCKET_NAME,
        "project_id": GCP_PROJECT_ID if GCP_PROJECT_ID else "Using default credentials",
        "error": gcs_error if not gcs_available else None,
        "message": "GCS reinitialized successfully" if result else f"GCS initialization failed: {gcs_error}"
    }

@myApp.post("/llm/upload")
async def upload_file(
    file: UploadFile = File(...),
    force_gcs: bool = Query(False, description="Force GCS upload, fail if not available")
):
    """
    Upload file to GCS or local storage
    
    Args:
        file: The file to upload
        force_gcs: If True, will fail if GCS is not available instead of falling back to local
    """
    # Read file content once to use for both GCS and local fallback
    file_content = await file.read()
    
    # Try GCS first if available
    if gcs_available and bucket:
        try:
            # Reset file pointer to beginning
            file.file.seek(0)
            
            blob = bucket.blob(file.filename)
            blob.upload_from_file(file.file, content_type=file.content_type)
            
            # Generate public URL (bucket must have public access or use signed URL)
            public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{file.filename}"
            print(f"✅ File uploaded to GCS: {public_url}")
            
            return {
                "message": "File uploaded to GCS successfully",
                "filename": file.filename,
                "url": public_url,
                "storage": "gcs"
            }
        except GoogleCloudError as e:
            error_code = getattr(e, 'code', None)
            error_msg = str(e)
            
            if force_gcs:
                # If force_gcs is True, don't fall back - return error
                raise HTTPException(
                    status_code=500,
                    detail=f"GCS upload failed ({error_code}): {error_msg}. "
                           f"Check billing and permissions. Error: {gcs_error or error_msg}"
                )
            
            # GCS upload failed - fall back to local storage
            print(f"⚠️ GCS upload failed ({error_code}): {error_msg}")
            print(f"   Falling back to local storage")
        except Exception as e:
            if force_gcs:
                raise HTTPException(status_code=500, detail=f"GCS upload failed: {str(e)}")
            print(f"⚠️ GCS upload failed ({e}), falling back to local storage")
    elif force_gcs:
        # GCS not available but force_gcs is True
        raise HTTPException(
            status_code=503,
            detail=f"GCS is not available. Error: {gcs_error or 'GCS not initialized'}. "
                   f"Please check your GCP configuration and billing."
        )
    
    # Fallback to local storage (either bucket is None or GCS upload failed)
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        print(f"✅ File saved locally: {file_path}")
        
        response = {
            "message": "File uploaded locally",
            "filename": file.filename,
            "path": str(file_path),
            "storage": "local"
        }
        
        # Add warning if GCS was attempted but failed
        if gcs_available and bucket:
            response["warning"] = "GCS upload failed, saved locally instead"
            response["gcs_error"] = gcs_error
        
        return response
    except Exception as e:
        print(f"❌ Local upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@myApp.get("/llm/files")
def list_files():
    """List all uploaded files"""
    files = []
    
    # List GCS files
    if bucket:
        try:
            blobs = bucket.list_blobs()
            files = [{"name": b.name, "url": b.public_url, "storage": "gcs"} for b in blobs]
        except Exception as e:
            print(f"Error listing GCS files: {e}")
    
    # Also list local files
    local_files = [{"name": f.name, "path": str(f), "storage": "local"} 
                   for f in UPLOAD_DIR.iterdir() if f.is_file()]
    
    return {"files": files + local_files}
    


@myApp.post("/api/chat")
def api_chat(request: ChatRequest):
    """Chat endpoint for frontend - accepts POST with messages"""
    try:
        # Get the last user message
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        print(f"Received query: {query}")
        
        # Check for multiple files first
        if request.fileNames and len(request.fileNames) > 0:
            print(f"📚 Processing {len(request.fileNames)} files: {request.fileNames}")
            rag_chain = llm().get_rag_chain_multi(request.fileNames)
            return EventSourceResponse(generate_rag_response(rag_chain, query))
        elif request.fileName:
            # Single file RAG mode (backward compatible)
            print(f"📄 Processing single file: {request.fileName}")
            rag_chain = llm().get_rag_chain(request.fileName)
            return EventSourceResponse(generate_rag_response(rag_chain, query))
        else:
            # Simple chat mode without document
            return EventSourceResponse(generate_chat_response(query))
    except Exception as e:
        print(f"Error in api_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@myApp.post("/api/chat/json")
def api_chat_json(request: ChatRequest):
    """Chat endpoint that returns JSON (for testing)"""
    try:
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        print(f"Received query: {query}")
        
        answer = llm().chat(query)
        return {"response": answer}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@myApp.get("/llm/chat")
def chat(query: str):
    """Simple chat endpoint without document (GET)"""
    return EventSourceResponse(generate_chat_response(query))

@myApp.get("/llm/search")
def search(query: str, fileName: str = None):
    """Search with optional document context (GET)"""
    if fileName:
        # RAG mode with document
        rag_chain = llm().get_rag_chain(fileName)
        return EventSourceResponse(generate_rag_response(rag_chain, query))
    else:
        # Simple chat mode without document
        return EventSourceResponse(generate_chat_response(query))

def generate_chat_response(input_query):
    """Generate response without document context"""
    try:
        answer = llm().chat(input_query)
        print(answer)
        for chunk in answer.split():
            yield f"{chunk} "
        yield "\n"
    except Exception as e:
        print(e)
        yield f"Error: {e}"

def generate_rag_response(rag_chain, input_query):
    """Generate response with document context (RAG)"""
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


# ==================== FINANCIAL ANALYSIS ENDPOINTS ====================

@myApp.post("/api/financial/analyze")
def analyze_financial_reports(request: FinancialAnalysisRequest):
    """
    Comprehensive financial report analysis
    - Extracts tables and data
    - Analyzes charts and visualizations
    - Provides financial insights
    """
    try:
        results = {
            "files": [],
            "combined_analysis": "",
            "tables": [],
            "charts": []
        }
        
        for fileName in request.fileNames:
            print(f"📊 Analyzing financial report: {fileName}")
            
            file_result = {
                "fileName": fileName,
                "tables": [],
                "charts": []
            }
            
            # Extract and analyze tables
            if request.extractTables:
                tables = extract_tables_from_pdf(fileName)
                for table in tables[:10]:  # Limit tables
                    table_analysis = analyze_table(table["markdown"]) if table["markdown"] else ""
                    file_result["tables"].append({
                        "page": table["page"],
                        "markdown": table["markdown"],
                        "analysis": table_analysis
                    })
                    results["tables"].append(table)
            
            # Analyze charts/images
            if request.analyzeCharts:
                images = extract_images_from_pdf(fileName, max_pages=5)
                for img in images:
                    chart_analysis = analyze_image_with_gemini(
                        img["base64"],
                        """Analyze this financial report page. Identify and explain:
                        1. Any charts, graphs or visualizations
                        2. Key financial data shown
                        3. Trends and patterns
                        4. Important metrics and their implications"""
                    )
                    file_result["charts"].append({
                        "page": img["page"],
                        "analysis": chart_analysis
                    })
                    results["charts"].append({"page": img["page"], "analysis": chart_analysis})
            
            results["files"].append(file_result)
        
        # Generate combined analysis if query provided
        if request.query:
            rag_chain = llm().get_financial_rag_chain(request.fileNames)
            response = rag_chain.invoke({"input": request.query})
            results["combined_analysis"] = response["answer"]
        
        # Format response based on requested format
        formatted_response = format_response(results, request.responseFormat)
        return formatted_response
        
    except Exception as e:
        print(f"Error in financial analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@myApp.post("/api/financial/tables")
def extract_financial_tables(fileNames: List[str]):
    """Extract all tables from financial reports"""
    try:
        all_tables = []
        for fileName in fileNames:
            tables = extract_tables_from_pdf(fileName)
            for table in tables:
                all_tables.append({
                    "file": fileName,
                    "page": table["page"],
                    "markdown": table["markdown"],
                    "data": table["data"]
                })
        
        return {"tables": all_tables, "count": len(all_tables)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@myApp.post("/api/financial/chat")
def financial_chat(request: ChatRequest):
    """
    Chat endpoint specialized for financial analysis
    Uses financial-specific prompts and analysis
    """
    try:
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1].content
        print(f"💰 Financial query: {query}")
        
        if request.fileNames and len(request.fileNames) > 0:
            rag_chain = llm().get_financial_rag_chain(request.fileNames)
            return EventSourceResponse(generate_rag_response(rag_chain, query))
        elif request.fileName:
            rag_chain = llm().get_financial_rag_chain([request.fileName])
            return EventSourceResponse(generate_rag_response(rag_chain, query))
        else:
            # Financial chat without documents
            return EventSourceResponse(generate_financial_chat_response(query))
    except Exception as e:
        print(f"Error in financial chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_financial_chat_response(input_query):
    """Generate response for financial questions without document context"""
    try:
        financial_prompt = f"""You are an expert financial analyst. Answer the following question 
        with detailed financial analysis and insights:
        
        {input_query}
        
        Provide specific metrics, ratios, and actionable insights where applicable."""
        
        answer = llm().chat(financial_prompt)
        print(answer)
        for chunk in answer.split():
            yield f"{chunk} "
        yield "\n"
    except Exception as e:
        print(e)
        yield f"Error: {e}"


# ==================== RESPONSE FORMAT ENDPOINTS ====================

@myApp.post("/api/financial/analyze/summary")
def analyze_financial_summary(request: FinancialAnalysisRequest):
    """Quick summary of financial reports"""
    request.responseFormat = "summary"
    return analyze_financial_reports(request)


@myApp.post("/api/financial/analyze/markdown")
def analyze_financial_markdown(request: FinancialAnalysisRequest):
    """Financial analysis in markdown format"""
    request.responseFormat = "markdown"
    return analyze_financial_reports(request)


@myApp.post("/api/financial/analyze/structured")
def analyze_financial_structured(request: FinancialAnalysisRequest):
    """Financial analysis in structured format for UI"""
    request.responseFormat = "structured"
    return analyze_financial_reports(request)


@myApp.post("/api/financial/compare")
def compare_financial_reports(request: FinancialAnalysisRequest):
    """
    Compare multiple financial reports
    Returns comparative analysis
    """
    try:
        if len(request.fileNames) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 files to compare")
        
        comparison_query = request.query or """
        Compare these financial reports and provide:
        1. Key differences in financial metrics
        2. Revenue/profit comparison
        3. Growth rate comparison
        4. Notable changes between periods
        5. Overall financial health comparison
        """
        
        # Get RAG chain for all files
        rag_chain = llm().get_financial_rag_chain(request.fileNames)
        response = rag_chain.invoke({"input": comparison_query})
        
        result = {
            "filesCompared": request.fileNames,
            "comparison": response["answer"],
            "format": "comparison"
        }
        
        # Add table comparison if requested
        if request.extractTables:
            all_tables = []
            for fileName in request.fileNames:
                tables = extract_tables_from_pdf(fileName)
                for table in tables[:5]:
                    all_tables.append({
                        "file": fileName,
                        "page": table["page"],
                        "markdown": table["markdown"]
                    })
            result["tables"] = all_tables
        
        return format_response(result, request.responseFormat)
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@myApp.post("/api/financial/metrics")
def extract_financial_metrics(fileNames: List[str], query: Optional[str] = None):
    """
    Extract key financial metrics from reports
    Returns structured metrics data
    """
    try:
        metrics_query = query or """
        Extract and list all key financial metrics from this document including:
        - Revenue figures
        - Profit margins
        - Growth rates
        - Key ratios (P/E, ROE, ROA, etc.)
        - Year-over-year changes
        
        Format each metric with its value and the period it represents.
        """
        
        rag_chain = llm().get_financial_rag_chain(fileNames)
        response = rag_chain.invoke({"input": metrics_query})
        
        return {
            "files": fileNames,
            "metrics": response["answer"],
            "format": "metrics"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))