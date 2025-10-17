import streamlit as st
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from io import BytesIO

def create_pdf(chat_history: list) -> BytesIO:
    """
    Generate PDF from chat history with properly formatted content and sources.
    chat_history should be a list of dicts with:
    - role: 'user' or 'assistant'
    - content: message text
    - contexts: list of context dicts (for assistant messages)
    """
    buffer = BytesIO()
    # Format timestamp with single-digit hours
    current_time = datetime.now(ZoneInfo("America/New_York"))
    formatted_time = current_time.strftime("%B %d, %Y %-I:%M %p EDT")
    
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles for better formatting
    user_style = ParagraphStyle(
        'UserStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor='#2E4057',
        spaceAfter=6
    )
    
    assistant_style = ParagraphStyle(
        'AssistantStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        leading=14
    )
    
    source_title_style = ParagraphStyle(
        'SourceTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor='#1a5490',
        spaceAfter=3
    )
    
    source_text_style = ParagraphStyle(
        'SourceText',
        parent=styles['Normal'],
        fontSize=8,
        textColor='#666666',
        spaceAfter=8,
        leftIndent=20
    )
    
    story = []
    
    # Header with date and time
    story.append(Paragraph(f"Chat History - {formatted_time}", styles["Title"]))
    story.append(Spacer(1, 20))
    
    # Process each message in chat history
    for msg_idx, msg in enumerate(chat_history):
        if msg["role"] == "user":
            # User message
            story.append(Paragraph(f"<b>User:</b>", user_style))
            story.append(Paragraph(msg["content"], assistant_style))
            story.append(Spacer(1, 12))
            
        elif msg["role"] == "assistant":
            # Assistant message
            story.append(Paragraph(f"<b>Assistant:</b>", user_style))
            
            # Process the content to handle markdown-style formatting
            content = msg["content"]
            
            # Split content into paragraphs and lists
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                # Check if this is a numbered list item
                if re.match(r'^\d+\.', line):
                    # Collect consecutive numbered items
                    list_items = []
                    while i < len(lines) and re.match(r'^\d+\.', lines[i].strip()):
                        item_text = re.sub(r'^\d+\.\s*', '', lines[i].strip())
                        list_items.append(item_text)
                        i += 1
                    
                    # Create numbered list
                    for idx, item in enumerate(list_items, 1):
                        story.append(Paragraph(f"{idx}. {item}", assistant_style))
                    story.append(Spacer(1, 6))
                    
                # Check if this is a bullet point
                elif line.startswith('- ') or line.startswith('* '):
                    # Collect consecutive bullet items
                    list_items = []
                    while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                        item_text = re.sub(r'^[-*]\s*', '', lines[i].strip())
                        list_items.append(item_text)
                        i += 1
                    
                    # Create bulleted list
                    for item in list_items:
                        story.append(Paragraph(f"• {item}", assistant_style))
                    story.append(Spacer(1, 6))
                    
                else:
                    # Regular paragraph
                    # Handle bold text
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    story.append(Paragraph(line, assistant_style))
                    i += 1
            
            story.append(Spacer(1, 10))
            
            # Add sources if available
            contexts = msg.get("contexts", [])
            if contexts:
                story.append(Paragraph("<b>Sources:</b>", styles["Heading2"]))
                story.append(Spacer(1, 6))
                
                for ctx in contexts:
                    title = extract_clean_title(ctx["file_name"])
                    pdf_url = create_direct_link(ctx["file_name"])
                    snippet = clean_chunk(ctx["chunk"])[:350] + ("..." if len(ctx["chunk"]) > 350 else "")
                    
                    # Source title with link
                    story.append(Paragraph(f'<link href="{pdf_url}">{title}</link>', source_title_style))
                    # Source snippet
                    story.append(Paragraph(snippet, source_text_style))
                
                story.append(Spacer(1, 15))
            else:
                story.append(Spacer(1, 15))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# Helper functions (keep these from original code)
def extract_clean_title(file_name: str) -> str:
    month_map = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December",
    }
    match = re.search(r"(\d{4})(\d{2})(\d{2})", file_name)
    if match:
        year, month, day = match.groups()
        date_str = f"{month_map.get(month, month)} {int(day)}, {year}"
    else:
        date_str = "Unknown Date"
    fname = file_name.lower()
    if "beigebook" in fname or "beigebook_" in fname:
        doc_type = "Beige Book"
    elif "longerungoals" in fname or "fomc_longerungoals" in fname:
        doc_type = "FOMC Longer-Run Goals"
    elif "presconf" in fname or "fomcpresconf" in fname:
        doc_type = "Press Conference"
    elif "projtabl" in fname or "fomcprojtabl" in fname:
        doc_type = "Projection Tables"
    elif "mprfullreport" in fname or "mpr" in fname:
        doc_type = "Monetary Policy Report"
    elif "monetary" in fname:
        doc_type = "Monetary Document"
    elif "financial-stability-report" in fname or "financial" in fname:
        doc_type = "Financial Stability Report"
    elif "minutes" in fname or "fomcminutes" in fname:
        doc_type = "FOMC Minutes"
    else:
        doc_type = "FOMC Document"
    return f"{doc_type} - {date_str}"


def create_direct_link(file_name: str) -> str:
    try:
        base = "https://www.federalreserve.gov"
        name = file_name.split("/")[-1]
        mapping = [
            (r"beigebook", f"{base}/monetarypolicy/files/"),
            (r"fomc_longerungoals", f"{base}/monetarypolicy/files/"),
            (r"fomcprojtabl", f"{base}/monetarypolicy/files/"),
            (r"fomcpresconf", f"{base}/mediacenter/files/"),
            (r"presconf", f"{base}/mediacenter/files/"),
            (r"monetary", f"{base}/monetarypolicy/files/"),
            (r"financial-stability-report", f"{base}/publications/files/"),
            (r"mprfullreport", f"{base}/monetarypolicy/files/"),
            (r"fomcminutes", f"{base}/monetarypolicy/files/"),
        ]
        lower = name.lower()
        for pattern, prefix in mapping:
            if pattern in lower:
                return prefix + name
        return f"{base}/monetarypolicy/files/{name}"
    except Exception as e:
        return f"https://www.federalreserve.gov/monetarypolicy/files/{file_name.split('/')[-1]}"


def clean_chunk(chunk: str) -> str:
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", chunk)
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
