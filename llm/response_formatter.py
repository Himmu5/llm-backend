"""
Response Formatter
Formats financial analysis responses in different formats for better readability
"""

from typing import Dict, Any, List
import json


def format_response(data: Dict[str, Any], format_type: str = "detailed") -> Dict[str, Any]:
    """
    Format the response based on the requested format type
    
    Formats:
    - summary: Key highlights only
    - detailed: Full analysis with all data
    - markdown: Markdown formatted for display
    - structured: Clean JSON structure for UI rendering
    """
    
    formatters = {
        "summary": format_summary,
        "detailed": format_detailed,
        "markdown": format_markdown,
        "structured": format_structured
    }
    
    formatter = formatters.get(format_type, format_detailed)
    return formatter(data)


def format_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summary format - Key highlights only
    """
    summary = {
        "format": "summary",
        "overview": {
            "filesAnalyzed": len(data.get("files", [])),
            "tablesFound": len(data.get("tables", [])),
            "chartsAnalyzed": len(data.get("charts", []))
        },
        "keyHighlights": [],
        "quickInsights": data.get("combined_analysis", "")[:500] + "..." if len(data.get("combined_analysis", "")) > 500 else data.get("combined_analysis", "")
    }
    
    # Extract key highlights from table analysis
    for file_data in data.get("files", []):
        for table in file_data.get("tables", [])[:3]:  # Top 3 tables
            if table.get("analysis"):
                # Extract first sentence as highlight
                first_sentence = table["analysis"].split('.')[0] + "."
                summary["keyHighlights"].append({
                    "source": f"{file_data['fileName']} - Page {table['page']}",
                    "highlight": first_sentence
                })
    
    return summary


def format_detailed(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detailed format - Full analysis with all data
    """
    return {
        "format": "detailed",
        "metadata": {
            "filesAnalyzed": len(data.get("files", [])),
            "totalTables": len(data.get("tables", [])),
            "totalCharts": len(data.get("charts", []))
        },
        "files": data.get("files", []),
        "combinedAnalysis": data.get("combined_analysis", ""),
        "allTables": data.get("tables", []),
        "allCharts": data.get("charts", [])
    }


def format_markdown(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Markdown format - Ready for display in markdown viewers
    """
    md_parts = []
    
    # Header
    md_parts.append("# 📊 Financial Analysis Report\n")
    
    # Overview
    md_parts.append("## Overview\n")
    md_parts.append(f"- **Files Analyzed:** {len(data.get('files', []))}")
    md_parts.append(f"- **Tables Found:** {len(data.get('tables', []))}")
    md_parts.append(f"- **Charts Analyzed:** {len(data.get('charts', []))}\n")
    
    # Combined Analysis
    if data.get("combined_analysis"):
        md_parts.append("## 💡 Key Insights\n")
        md_parts.append(data["combined_analysis"] + "\n")
    
    # Tables Section
    if data.get("files"):
        md_parts.append("## 📋 Financial Tables\n")
        for file_data in data["files"]:
            md_parts.append(f"### {file_data['fileName']}\n")
            for table in file_data.get("tables", []):
                md_parts.append(f"#### Table (Page {table['page']})\n")
                if table.get("markdown"):
                    md_parts.append(table["markdown"] + "\n")
                if table.get("analysis"):
                    md_parts.append(f"**Analysis:** {table['analysis']}\n")
    
    # Charts Section
    if data.get("charts"):
        md_parts.append("## 📈 Chart Analysis\n")
        for chart in data["charts"]:
            md_parts.append(f"### Page {chart['page']}\n")
            md_parts.append(chart.get("analysis", "") + "\n")
    
    return {
        "format": "markdown",
        "content": "\n".join(md_parts),
        "raw": data
    }


def format_structured(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structured format - Clean JSON structure for UI rendering
    Organized by sections for easy frontend consumption
    """
    structured = {
        "format": "structured",
        "sections": []
    }
    
    # Overview Section
    structured["sections"].append({
        "id": "overview",
        "title": "Overview",
        "type": "stats",
        "data": {
            "stats": [
                {"label": "Files Analyzed", "value": len(data.get("files", [])), "icon": "file"},
                {"label": "Tables Found", "value": len(data.get("tables", [])), "icon": "table"},
                {"label": "Charts Analyzed", "value": len(data.get("charts", [])), "icon": "chart"}
            ]
        }
    })
    
    # Key Insights Section
    if data.get("combined_analysis"):
        structured["sections"].append({
            "id": "insights",
            "title": "Key Insights",
            "type": "text",
            "data": {
                "content": data["combined_analysis"]
            }
        })
    
    # Financial Metrics Section (extracted from analysis)
    metrics = extract_metrics_from_analysis(data)
    if metrics:
        structured["sections"].append({
            "id": "metrics",
            "title": "Financial Metrics",
            "type": "metrics",
            "data": {
                "metrics": metrics
            }
        })
    
    # Tables Section
    tables_section = {
        "id": "tables",
        "title": "Financial Tables",
        "type": "tables",
        "data": {
            "tables": []
        }
    }
    
    for file_data in data.get("files", []):
        for table in file_data.get("tables", []):
            tables_section["data"]["tables"].append({
                "source": file_data["fileName"],
                "page": table["page"],
                "content": table.get("markdown", ""),
                "analysis": table.get("analysis", "")
            })
    
    if tables_section["data"]["tables"]:
        structured["sections"].append(tables_section)
    
    # Charts Section
    charts_section = {
        "id": "charts",
        "title": "Visual Analysis",
        "type": "charts",
        "data": {
            "analyses": []
        }
    }
    
    for chart in data.get("charts", []):
        charts_section["data"]["analyses"].append({
            "page": chart["page"],
            "analysis": chart.get("analysis", "")
        })
    
    if charts_section["data"]["analyses"]:
        structured["sections"].append(charts_section)
    
    return structured


def extract_metrics_from_analysis(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract financial metrics from the analysis text
    """
    metrics = []
    
    # Common financial metrics to look for
    metric_keywords = [
        ("Revenue", "$", "revenue"),
        ("Net Income", "$", "profit"),
        ("EBITDA", "$", "ebitda"),
        ("Gross Margin", "%", "margin"),
        ("ROE", "%", "return"),
        ("ROA", "%", "return"),
        ("Debt-to-Equity", "ratio", "leverage"),
        ("P/E Ratio", "x", "valuation"),
        ("EPS", "$", "earnings"),
        ("Operating Margin", "%", "margin")
    ]
    
    combined_text = data.get("combined_analysis", "")
    
    # Add file analyses
    for file_data in data.get("files", []):
        for table in file_data.get("tables", []):
            combined_text += " " + table.get("analysis", "")
    
    # Simple extraction (could be enhanced with NLP)
    for metric_name, unit, category in metric_keywords:
        if metric_name.lower() in combined_text.lower():
            metrics.append({
                "name": metric_name,
                "unit": unit,
                "category": category,
                "found": True
            })
    
    return metrics[:10]  # Limit to top 10


def format_for_chat(analysis_result: Dict[str, Any]) -> str:
    """
    Format analysis result as a chat-friendly response
    """
    parts = []
    
    # Summary
    parts.append("📊 **Financial Analysis Complete**\n")
    parts.append(f"Analyzed {len(analysis_result.get('files', []))} file(s)\n")
    
    # Key findings
    if analysis_result.get("combined_analysis"):
        parts.append("\n💡 **Key Findings:**")
        parts.append(analysis_result["combined_analysis"])
    
    # Table highlights
    table_count = sum(len(f.get("tables", [])) for f in analysis_result.get("files", []))
    if table_count > 0:
        parts.append(f"\n📋 Found {table_count} financial table(s)")
    
    # Chart highlights
    chart_count = len(analysis_result.get("charts", []))
    if chart_count > 0:
        parts.append(f"\n📈 Analyzed {chart_count} chart(s)/visualization(s)")
    
    return "\n".join(parts)

