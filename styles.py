from types import SimpleNamespace
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, ListStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import portrait, A4

sample_styles = getSampleStyleSheet()

TRANSPARENT = '#FFFFFF00'

FONT_TABLE = {
    '': 'Courier',
    'b': 'Courier-Bold',
    'i': 'Courier-Oblique',
    'bi': 'Courier-BoldOblique',
}

FONT_GRAPH = {
    '': 'Helvetica',
    'b': 'Helvetica-Bold',
    'i': 'Helvetica-Oblique',
    'bi': 'Helvetica-BoldOblique'
}

FONT_TABLE_SIZE = 8
INDENT_SIZE = 0.5*cm

LABEL_WRAP_LENGTH = 25       # in graph labels

TABLE_GRID_WIDTH = 0.2
TABLE_BORDER_WIDTH = 1.5

# ======== REPORTLAB STYLES ======== 

# paragraph styles
class PSTYLE(SimpleNamespace):
    N = ParagraphStyle('pstyle', sample_styles['Normal'], 
        wordWrap='CJK',
        fontSize=FONT_TABLE_SIZE,
        fontName=FONT_TABLE['']
    )
    """normal"""

    B = ParagraphStyle('pstyle_b', N, 
        fontName=FONT_TABLE['b']
    )
    """bold"""

    I = ParagraphStyle('pstyle_i', N, 
        fontName=FONT_TABLE['i']
    )
    """italic"""

    BI = ParagraphStyle('pstyle_bi', N, 
        fontName=FONT_TABLE['bi']
    ) 
    """bold italic"""

    INDENT = ParagraphStyle('pstyle_indent', N, 
        leftIndent=INDENT_SIZE
    )

# list styles
class LSTYLE(SimpleNamespace):
    # for some reason, ListStyle doesn't work so we use dicts for now
    BULLET = {      
        'bulletType': 'bullet', 
        'bulletFontSize': FONT_TABLE_SIZE,
        'leftIndent': INDENT_SIZE
    }


TABLE_ATTRIBS = {
    'colWidths': [1.5*cm, 16*cm, 1*cm]
}

class TSTYLE(SimpleNamespace):
    BASE = [
        # all
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.2*cm),
        ('TOPPADDING', (0, 0), (-1, -1), 0.2*cm),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('GRID', (0, 0), (-1, -1), TABLE_GRID_WIDTH, colors.black),
        ('BOX', (0, 0), (-1, -1), TABLE_BORDER_WIDTH, colors.black),

        # left column
        ('FONTNAME', (0, 0), (0, -1), FONT_TABLE['b']),
        ('FONTSIZE', (0, 0), (0, -1), FONT_TABLE_SIZE),

        # right columns
        ('FONTNAME', (1, 0), (-1, -1), FONT_TABLE['']),
        ('FONTSIZE', (1, 0), (-1, -1), FONT_TABLE_SIZE),

        # right column
        ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
    ]

    STEP = [
        ('LINEABOVE', TABLE_BORDER_WIDTH, colors.black)
    ]

DOC_STYLE = {
    'pagesize':     portrait(A4),
    'topMargin':    1*cm,
    'bottomMargin': 1*cm,
    'leftMargin':   1*cm,
    'rightMargin':  1*cm,
}


# ======== GRAPHVIZ STYLES ======== 


class ATTR(SimpleNamespace):
    class GRAPH(SimpleNamespace):
        BASE = {
            'graph': {
                'compound': 'true',
                'rankdir': 'TB',
                # 'overlap': 'true',
                'fontname': FONT_GRAPH[''],
                # 'splines': 'ortho',
                'ranksep': '0.7',
                'nodesep': '0.1',
                'searchsize': '0'
            },
            'node': {
                'fontname': FONT_GRAPH[''],
            },
            'edge': {
                'fontname': FONT_GRAPH[''],
                'tailport': 's',
                'headport': 'n',
            }
        }

        CLEAR = {
            'graph': {
                **BASE['graph'],
                'color': TRANSPARENT,
                'fillcolor': TRANSPARENT,
                'label': '',
            }
        }

        STEP = {
            'graph': {
                **BASE['graph'],
                'style': 'filled', 
                'color': 'grey',
                'fillcolor': '#00ffff10',
                'labeljust': 'l',
            },
        }

    TERMINAL = {
        'shape': 'box', 
        'style': 'filled',
        'fillcolor': '#e6e6e6',
        'color': '#000000',
        'fontname': FONT_GRAPH['b'],
    }

    STEP = {
        'shape': 'box',
        'style': 'filled',
        'fillcolor': "#00ffff",
        'color': '#0000ff'
    }

    POINT = {
        'shape': 'point',
        'height': '0'
    }

    FLOW = {
        'style': 'filled',
        'fillcolor': "#ffff00",
        'color': '#fd8b42'
    }

    IF = {
        **FLOW,
        'shape': 'diamond',
    }

    CASE = {
        **FLOW,
        'shape': 'hexagon',
    }

    WAIT = {
        'shape': 'box',
        'style': 'filled,rounded',
        'fillcolor': "#EEEEEE",
        'color': '#777777'
    }