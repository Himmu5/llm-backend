"""
Financial Document Processor
Handles extraction and analysis of financial reports including:
- Text content
- Tables
- Charts and visualizations
"""

import os
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import tempfile
import requests

# PDF Processing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not installed. Table extraction disabled.")

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not installed. Image extraction disabled.")

from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "himanshu-rag")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ROOT = Path(__file__).parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"


def get_pdf_path(fileName: str) -> str:
    """Get the full path or URL for a PDF file"""
    if fileName.startswith(('http://', 'https://')):
        return fileName
    elif fileName.startswith('/'):
        return fileName
    else:
        uploads_path = UPLOADS_DIR / fileName
        docs_path = PROJECT_ROOT / "docs" / fileName
        
        if uploads_path.exists():
            return str(uploads_path)
        elif docs_path.exists():
            return str(docs_path)
        else:
            encoded_filename = quote(fileName, safe='')
            return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{encoded_filename}"


def download_pdf_if_url(file_path: str) -> str:
    """Download PDF if it's a URL, return local path"""
    if file_path.startswith(('http://', 'https://')):
        response = requests.get(file_path)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(response.content)
            return tmp.name
    return file_path


def extract_tables_from_pdf(fileName: str) -> List[Dict[str, Any]]:
    """Extract tables from a PDF file using pdfplumber"""
    if not PDFPLUMBER_AVAILABLE:
        return []
    
    file_path = get_pdf_path(fileName)
    local_path = download_pdf_if_url(file_path)
    
    tables = []
    try:
        with pdfplumber.open(local_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 0:
                        # Convert table to markdown format
                        markdown_table = convert_table_to_markdown(table)
                        tables.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "data": table,
                            "markdown": markdown_table
                        })
        
        print(f"📊 Extracted {len(tables)} tables from {fileName}")
    except Exception as e:
        print(f"Error extracting tables: {e}")
    
    return tables


def convert_table_to_markdown(table: List[List[str]]) -> str:
    """Convert a table to markdown format"""
    if not table or len(table) == 0:
        return ""
    
    # Clean up None values
    clean_table = [[str(cell) if cell else "" for cell in row] for row in table]
    
    # Create markdown
    lines = []
    
    # Header row
    if len(clean_table) > 0:
        lines.append("| " + " | ".join(clean_table[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(clean_table[0])) + " |")
    
    # Data rows
    for row in clean_table[1:]:
        # Pad row if necessary
        while len(row) < len(clean_table[0]):
            row.append("")
        lines.append("| " + " | ".join(row[:len(clean_table[0])]) + " |")
    
    return "\n".join(lines)


def extract_images_from_pdf(fileName: str, max_pages: int = 5) -> List[Dict[str, Any]]:
    """Extract images (page renders) from PDF for chart analysis"""
    if not PDF2IMAGE_AVAILABLE:
        return []
    
    file_path = get_pdf_path(fileName)
    local_path = download_pdf_if_url(file_path)
    
    images = []
    try:
        # Convert PDF pages to images
        pil_images = convert_from_path(local_path, first_page=1, last_page=max_pages, dpi=150)
        
        for page_num, img in enumerate(pil_images, 1):
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.standard_b64encode(buffered.getvalue()).decode()
            
            images.append({
                "page": page_num,
                "base64": img_base64,
                "width": img.width,
                "height": img.height
            })
        
        print(f"📸 Extracted {len(images)} page images from {fileName}")
    except Exception as e:
        print(f"Error extracting images: {e}")
    
    return images


def analyze_image_with_gemini(image_base64: str, prompt: str) -> str:
    """Analyze an image using Gemini Vision"""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not configured"
    
    try:
        # Use Gemini with vision capabilities
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            ]
        )
        
        response = model.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"Error analyzing image: {str(e)}"


def analyze_financial_document(fileName: str) -> Dict[str, Any]:
    """
    Comprehensive financial document analysis
    Returns text, tables, and image analysis
    """
    result = {
        "fileName": fileName,
        "tables": [],
        "table_analysis": [],
        "chart_analysis": [],
        "summary": ""
    }
    
    # Extract tables
    tables = extract_tables_from_pdf(fileName)
    result["tables"] = tables
    
    # Analyze each table
    for table in tables[:5]:  # Limit to first 5 tables
        if table["markdown"]:
            analysis = analyze_table(table["markdown"])
            result["table_analysis"].append({
                "page": table["page"],
                "analysis": analysis
            })
    
    # Extract and analyze images (for charts)
    images = extract_images_from_pdf(fileName, max_pages=3)
    for img in images:
        chart_prompt = """Analyze this page from a financial report. 
        Identify any charts, graphs, or visualizations and explain:
        1. What type of visualization is shown
        2. Key data points and trends
        3. Financial implications"""
        
        analysis = analyze_image_with_gemini(img["base64"], chart_prompt)
        result["chart_analysis"].append({
            "page": img["page"],
            "analysis": analysis
        })
    
    return result


def analyze_table(markdown_table: str) -> str:
    """Analyze a financial table using LLM"""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not configured"
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY
        )
        
        prompt = f"""Analyze this financial table and provide:
1. Summary of key figures
2. Notable trends or patterns
3. Key financial metrics visible
4. Important takeaways

Table:
{markdown_table}
"""
        
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error analyzing table: {str(e)}"


# Export functions
__all__ = [
    'extract_tables_from_pdf',
    'extract_images_from_pdf', 
    'analyze_image_with_gemini',
    'analyze_financial_document',
    'analyze_table',
    'convert_table_to_markdown'
]

