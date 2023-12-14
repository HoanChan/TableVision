import xml.etree.ElementTree as ET

def cells_to_html(cells):

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = cell['row']

        attrib = {}
        colspan = cell['col_span']
        if colspan > 1:
            attrib['colspan'] = str(colspan)
        rowspan = cell['row_span']
        if rowspan > 1:
            attrib['rowspan'] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if current_row == 0:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell['cell text']

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))

def createHTML(image_path, html, show_image = True):
    css = """
    table {
    border-collapse: collapse;
    border-spacing: 0;
    width: 100%;
    font-family: sans-serif;
    }

    th, td {
    padding: 8px;
    text-align: center;
    vertical-align: middle;
    }

    th {
    font-weight: bold;
    font-size: 1.2em;
    }

    tr:nth-child(even) {
    background-color: rgba(68, 68, 68, 0.2);
    }

    table th, table td {
    border: 1px solid rgba(68, 68, 68, 0.5);
    }
    """
    body = """
    <div style="display: flex;">
    <div style="flex: 1;">
        <img src='""" + image_path + """' alt="Ảnh" style="width: 90%;">
    </div>
    <div style="flex: 1;">""" + html + """</div>
    </div>
    """ if show_image else html
    new_html = '<head><style>'+ css +'</style></head><body>'+ body + '</body>'
    return new_html
