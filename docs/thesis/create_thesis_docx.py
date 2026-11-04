#!/usr/bin/env python3
"""
Convert khung-do-an.md into Khung_do_an.docx using the OpenXML template from FinanceForecast.
"""

import os
import re
import zipfile
import xml.sax.saxutils as saxutils

def escape_xml(text):
    return saxutils.escape(text)

def make_run(text, bold=False, italic=False, code=False):
    rpr = []
    if bold:
        rpr.append('<w:b/>')
    if italic:
        rpr.append('<w:i/>')
    if code:
        rpr.append('<w:rFonts w:ascii="JetBrains Mono" w:hAnsi="JetBrains Mono"/>')
        rpr.append('<w:color w:val="0369A1"/>')
    rpr_str = f'<w:rPr>{"".join(rpr)}</w:rPr>' if rpr else ''
    escaped_text = escape_xml(text)
    return f'<w:r>{rpr_str}<w:t xml:space="preserve">{escaped_text}</w:t></w:r>'

def parse_inline_formatting(text):
    """Parses bold, italic, inline code in markdown text and returns OpenXML runs."""
    # Pattern to match **bold**, *italic*, and `code`
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    runs = []
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**') and len(token) >= 4:
            runs.append(make_run(token[2:-2], bold=True))
        elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
            runs.append(make_run(token[1:-1], italic=True))
        elif token.startswith('`') and token.endswith('`') and len(token) >= 2:
            runs.append(make_run(token[1:-1], code=True))
        else:
            runs.append(make_run(token))
    return "".join(runs)

def make_paragraph(content_runs, style="BodyText", align=None, page_break_before=False):
    pPr = []
    if style:
        pPr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        pPr.append(f'<w:jc w:val="{align}"/>')
    if page_break_before:
        pPr.append('<w:pageBreakBefore/>')
    pPr_str = f'<w:pPr>{"".join(pPr)}</w:pPr>' if pPr else ''
    return f'<w:p>{pPr_str}{content_runs}</w:p>\n'

def make_table(rows):
    """Constructs a clean OpenXML table from list of lists of strings."""
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    grid_cols = "".join(['<w:gridCol w:w="4500"/>'] * col_count)
    
    xml_rows = []
    for row_idx, r in enumerate(rows):
        is_header = (row_idx == 0)
        tc_xmls = []
        for cell in r:
            cell_runs = parse_inline_formatting(cell.strip())
            p_style = "FirstParagraph" if is_header else "Compact"
            p_xml = make_paragraph(cell_runs, style=p_style)
            
            tcPr = ['<w:tcW w:w="0" w:type="auto"/>']
            if is_header:
                tcPr.append('<w:shd w:fill="F1F5F9" w:val="clear"/>')
            tcPr_str = f'<w:tcPr>{"".join(tcPr)}</w:tcPr>'
            tc_xmls.append(f'<w:tc>{tcPr_str}{p_xml}</w:tc>')
            
        # fill missing cells
        while len(tc_xmls) < col_count:
            tc_xmls.append('<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr><w:p/></w:tc>')
            
        xml_rows.append(f'<w:tr>{"".join(tc_xmls)}</w:tr>')
        
    tbl_pr = (
        '<w:tblPr>'
        '<w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        '<w:insideV w:val="none"/>'
        '<w:left w:val="none"/>'
        '<w:right w:val="none"/>'
        '</w:tblBorders>'
        '</w:tblPr>'
    )
    return f'<w:tbl>{tbl_pr}<w:tblGrid>{grid_cols}</w:tblGrid>{"".join(xml_rows)}</w:tbl>\n'

def md_to_docx_body(md_content):
    lines = md_content.split('\n')
    body_elements = []
    
    i = 0
    in_table = False
    table_rows = []
    in_center = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check center tags
        if '<div align="center">' in line:
            in_center = True
            i += 1
            continue
        if '</div>' in line:
            in_center = False
            i += 1
            continue
            
        # Check horizontal rule
        if line.startswith('---'):
            if in_table:
                body_elements.append(make_table(table_rows))
                table_rows = []
                in_table = False
            # add subtle spacing
            body_elements.append(make_paragraph('', style='BodyText'))
            i += 1
            continue
            
        # Check table
        if line.startswith('|') and line.endswith('|'):
            # check if it's separator row
            if re.match(r'^\|[\s:-]+\|$', line.replace('|', '| |')):
                i += 1
                continue
            cells = [c.strip() for c in line[1:-1].split('|')]
            table_rows.append(cells)
            in_table = True
            i += 1
            continue
        else:
            if in_table:
                body_elements.append(make_table(table_rows))
                table_rows = []
                in_table = False
                
        if not line or line in ('<br/>', '<br>', '<br />'):
            i += 1
            continue
            
        align = "center" if in_center else None
        
        # Headings
        if line.startswith('# '):
            body_elements.append(make_paragraph(parse_inline_formatting(line[2:]), style='Heading1', align=align))
        elif line.startswith('## '):
            body_elements.append(make_paragraph(parse_inline_formatting(line[3:]), style='Heading2', align=align))
        elif line.startswith('### '):
            body_elements.append(make_paragraph(parse_inline_formatting(line[4:]), style='Heading3', align=align))
        elif line.startswith('#### '):
            body_elements.append(make_paragraph(parse_inline_formatting(line[5:]), style='Heading4', align=align))
        elif line.startswith('> '):
            # Blockquote
            body_elements.append(make_paragraph(parse_inline_formatting(line[2:]), style='BlockText'))
        elif line.startswith('- ') or line.startswith('* '):
            bullet_run = make_run('•  ', bold=True)
            body_elements.append(make_paragraph(bullet_run + parse_inline_formatting(line[2:]), style='Compact'))
        elif re.match(r'^\d+\.\s', line):
            num_match = re.match(r'^(\d+\.\s)(.*)', line)
            num_str = num_match.group(1)
            rest_str = num_match.group(2)
            num_run = make_run(num_str, bold=True)
            body_elements.append(make_paragraph(num_run + parse_inline_formatting(rest_str), style='Compact'))
        else:
            body_elements.append(make_paragraph(parse_inline_formatting(line), style='BodyText', align=align))
            
        i += 1
        
    if in_table and table_rows:
        body_elements.append(make_table(table_rows))
        
    return "".join(body_elements)

def build_docx(template_path, md_path, output_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    new_body_content = md_to_docx_body(md_text)
    
    with zipfile.ZipFile(template_path, 'r') as zin:
        orig_doc_xml = zin.read('word/document.xml').decode('utf-8')
        
        # Extract w:sectPr
        sect_match = re.search(r'(<w:sectPr.*?</w:sectPr>)', orig_doc_xml, re.DOTALL)
        sect_xml = sect_match.group(1) if sect_match else '<w:sectPr/>'
        
        # Extract root tag and namespaces
        root_match = re.search(r'(<w:document[^>]*>)', orig_doc_xml)
        root_tag = root_match.group(1) if root_match else '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        
        new_doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'{root_tag}\n'
            f'<w:body>\n{new_body_content}\n{sect_xml}\n</w:body>\n'
            '</w:document>'
        )
        
        # Write new docx
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == 'word/document.xml':
                    zout.writestr(item, new_doc_xml.encode('utf-8'))
                else:
                    zout.writestr(item, zin.read(item.filename))
                    
    print(f"✅ Generated DOCX successfully: {output_path}")

if __name__ == '__main__':
    template_docx = '/Users/dxng/Documents/Codes/FinanceForecast/docs/thesis/Khung_do_an.docx'
    md_file = '/Users/dxng/Documents/Codes/WeatherForecast/docs/thesis/khung-do-an.md'
    output_docx = '/Users/dxng/Documents/Codes/WeatherForecast/docs/thesis/Khung_do_an.docx'
    
    build_docx(template_docx, md_file, output_docx)
