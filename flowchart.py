from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Dict, List, NamedTuple, Optional, Type, TypeVar, Union, get_origin, get_args
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pluto_parser.parser import PlutoParser
from lark import Token, Tree, v_args
from lark.visitors import Interpreter, visit_children_decor
import graphviz
import html
import re
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, ListStyle
from reportlab.lib.pagesizes import portrait, A4
from reportlab.platypus import (
    SimpleDocTemplate, 
    Table, TableStyle, 
    Paragraph, ListFlowable, 
    Flowable, PageTemplate, 
    PageBreak, NextPageTemplate, Frame
)
from reportlab.graphics.shapes import Drawing
from reportlab.lib.units import cm
from svglib.svglib import svg2rlg

styles = getSampleStyleSheet()


FONT_TABLE = {
    '': 'Courier',
    'b': 'Courier-Bold',
    'i': 'Courier-Oblique',
    'bi': 'Courier-BoldOblique',
}

FONT_GRAPH = {
    '': 'helvetica'
}

FONT_TABLE_SIZE = 8
INDENT_SIZE = 0.5*cm

LINE_WRAP_LENGTH = 25


# REPORTLAB STYLES
pstyle = ParagraphStyle('pstyle', styles['Normal'], 
    wordWrap='CJK',
    fontSize=FONT_TABLE_SIZE,
    fontName=FONT_TABLE['']
)

pstyle_b = ParagraphStyle('pstyle_b', pstyle, 
    fontName=FONT_TABLE['b']
)

pstyle_i = ParagraphStyle('pstyle_i', pstyle, 
    fontName=FONT_TABLE['i']
)

pstyle_bi = ParagraphStyle('pstyle_bi', pstyle, 
    fontName=FONT_TABLE['bi']
)

pstyle_indent = ParagraphStyle('pstyle_indent', pstyle, 
    leftIndent=INDENT_SIZE
)

lstyle_bullet = {
    'bulletType': 'bullet', 
    'bulletFontSize': FONT_TABLE_SIZE,
    'leftIndent': INDENT_SIZE
}


# GRAPHVIZ STYLES
ATTR_TERMINAL = {
    'shape': 'box', 
    'style': 'filled',
    'fillcolor': '#e6e6e6',
    'color': '#000000',
    'fontname': 'helvetica-bold',
}

ATTR_STEP = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': "#00ffff",
    'color': '#0000ff'
}

ATTR_POINT = {
    'shape': 'point',
    'height': '0'
}

ATTR_GRAPH_BASE = {
    'graph': {
        'compound': 'true',
        'rankdir': 'TB',
        # 'overlap': 'true',
        'fontname': 'helvetica',

        # 'splines': 'ortho',
        'ranksep': '0.7',
        # 'nodesep': '2'
        'searchsize': '0'
    },
    'node': {
        'fontname': 'helvetica',
    },
    'edge': {
        'fontname': 'helvetica',
        'tailport': 's',
        'headport': 'n',
    }
}

ATTR_GRAPH_CLEAR = {
    'graph': {
        **ATTR_GRAPH_BASE['graph'],
        'color': '#ffffff',
        'fillcolor': '#ffffff',
        'label': '',
    }
}

ATTR_GRAPH_STEP = {
    'graph': {
        **ATTR_GRAPH_BASE['graph'],
        'style': 'filled', 
        'color': 'grey',
        'fillcolor': 'white',
        'labeljust': 'l'
    },
}

ATTR_FLOW = {
    'style': 'filled',
    'fillcolor': "#ffff00",
    'color': '#fd8b42'
}

ATTR_IF = {
    **ATTR_FLOW,
    'shape': 'diamond',
}

ATTR_CASE = {
    **ATTR_FLOW,
    'shape': 'hexagon',
}

ATTR_WAIT = {
    'shape': 'box',
    'style': 'filled,rounded',
    'fillcolor': "#EEEEEE",
    'color': '#777777'
}



def flatten_list(input_list: list, keep_nones=False):
    flat_list = []
    for entry in input_list:
        if type(entry) == list:
            flat_list.extend(flatten_list(entry))
        elif entry is None and not keep_nones:
            continue
        else:
            flat_list.append(entry)
    return flat_list

FilterType = TypeVar('FilterType')
def filter_list(input_list: list, filter_type: Type[FilterType]) -> List[FilterType]:
    flat_list = flatten_list(input_list)
    filtered_list = []
    for entry in flat_list:
        if type(entry) == filter_type or isinstance(entry, filter_type):
            filtered_list.append(entry)
    return filtered_list

def get_child(tree: Optional[Tree], name: str) -> Union[Tree, Token, None]:
    if not tree:
        return None
    for child in tree.children:
        if isinstance(child, Tree) and child.data == name:
            return child
        elif isinstance(child, Token) and child.type == name:
            return child
        
def get_children(tree: Optional[Tree], name: Union[str, List[str]]) -> List[Union[Tree, Token]]:
    children = []
    names = []
    if type(name) == str:
        names = [name]
    elif type(name) == list:
        names = name

    if not tree:
        return []
    for child in tree.children:
        if (isinstance(child, Tree) and child.data in names
            or isinstance(child, Token) and child.type in names
        ):
            children.append(child)
    return children

def get_unique_name(node: Union[Tree, Token], suffix: str='') -> str:
    name = ''
    if isinstance(node, Tree):
        name = f"{node.data}_{node.meta.line}_{node.meta.column}"
    if isinstance(node, Token):
        name = f"{node.type}_{node.line}_{node.column}"
    
    if suffix:
        name += f'_{suffix}'

    return name
    
def apply_graph_attr(graph: graphviz.Digraph, style: dict={}):
    for target in style:
        graph.attr(target, **style[target])

def get_type(node: Union[Tree, Token, str]) -> str:
    if isinstance(node, Tree):
        if isinstance(node.data, Token):
            return node.data.value
        else:
            return node.data
    if isinstance(node, Token):
        return node.type
    if type(node):
        return node

def word_wrap(string: str, max_length: int=LINE_WRAP_LENGTH) -> str:
    # replace newlines, and multiple whitespaces with single whitespace
    # and remove leading and trailing whitespaces
    string = re.sub(r'[\n\s\r]+', " ", string).strip()

    line_length = 0
    output = ""
    for word in string.split(" "):
        if line_length == 0:
            output += word
            line_length += len(word)
        elif line_length + len(word) + 1 > max_length:
            output += ('\\n' + word)
            line_length = len(word)
        else:
            output += (" " + word)
            line_length += len(word) + 1

    return output




class GraphNode(NamedTuple):
    entry: str
    exit: str

@dataclass
class StepDetails:
    text: str=''
    declaration: str=''


@dataclass
class Step:
    node_name: str
    header: str = ''
    details: str = ''
    number_str: str = ''

@dataclass
class GoToStep:
    node_name: str
    label: str = ''

class PlutoFlowchart(Interpreter):
    code: str = ""
    node_followers: Dict[str, List[GoToStep]] = {}

    step_counter = 0
    step_num_lookup: Dict[str, str] = {}

    def __init__(self, graph: graphviz.Digraph=None):
        super().__init__()

        self.steps: Dict[str, Step] = {}

        if graph:
            self.g = graph
        else:
            self.g = graphviz.Digraph('G', format='svg')
            apply_graph_attr(self.g, ATTR_GRAPH_BASE)

    def parse(file: str) -> Tree:
        with open(file) as f:
            PlutoFlowchart.code = f.read()

        return PlutoParser.parse(PlutoFlowchart.code)

    def create_flowchart(self, pluto_file:str, output_file: str):
        tree = PlutoFlowchart.parse(pluto_file)
        self.visit(tree)
        self.g.render(output_file)

    def node(self, label: str, node: Union[Tree, Token]=None, 
                    name: str=None, html_label: bool=False,
                    **node_attributes) -> GraphNode:
        if not name and not node:
            return
        node_name = get_unique_name(node, name) if node else name

        node_label = label

        if not html_label:
            node_label = html.escape(word_wrap(label))

        self.g.node(node_name, label=node_label, **node_attributes)
        
        return GraphNode(node_name, node_name)
    
    def step(self, header: str, details: str='', node: Union[Tree, Token]=None, 
             node_name: str='', **node_attributes) -> GraphNode:
            
        if not node_name and not node:
            return
        node_name = get_unique_name(node, node_name) if node else node_name

        number_str = str(PlutoFlowchart.step_counter)
        PlutoFlowchart.step_counter += 1

        step = Step(node_name, header, details, number_str)

        self.steps[step.node_name] = step
        PlutoFlowchart.step_num_lookup[step.node_name] = step.number_str
            

        escaped_label = html.escape(word_wrap(step.header)).replace("\\n", "<BR/>")
        node_label = (
            '<<TABLE border="0" cellborder="0" cellspacing="0" ALIGN="CENTER">'
            f'<TR><TD ALIGN="LEFT" VALIGN="TOP"><I><B>{step.number_str}</B></I>'
            f'</TD><TD ALIGN="CENTER" >{escaped_label}</TD></TR>'
            '</TABLE>>'
        )
        tooltip = f'Step {step.number_str}'

        return self.node(node_label, name=node_name, html_label=True, **{'tooltip': tooltip, **node_attributes})
    
    def edges_sequential(self, nodes: List[GraphNode], entry: Union[str, GraphNode]=None, exit: Union[str, GraphNode]=None, 
                         entry_attr: dict={}, exit_attr: dict={}, **edge_attributes):
        node_list = filter_list(nodes, GraphNode)
        if entry:
            entry_node = entry
            if type(entry) == str:
                entry_node = GraphNode(None, entry)
            node_list = [entry_node] + node_list
        if exit:
            exit_node = exit
            if type(exit) == str:
                exit_node = GraphNode(exit, None)
            node_list = node_list + [exit_node]

        for node, next_node in zip(node_list, node_list[1:]):
            edge_attr = edge_attributes
            if node == node_list[0]:
                edge_attr = {**edge_attr, **entry_attr}
            if next_node == node_list[-1]:
                edge_attr = {**edge_attr, **exit_attr}
            if len(node_list) == 2:
                if 'tailport' in entry_attr:
                    edge_attr['tailport'] = entry_attr['tailport']
            self.edge(node.exit, next_node.entry, **edge_attr)

        return node_list
    

    def edges_parallel(self, nodes: List[GraphNode], entry: Union[str, GraphNode]=None, exit: Union[str, GraphNode]=None, 
                         entry_attr: dict={}, exit_attr: dict={}, **edge_attributes):
        node_list = filter_list(nodes, GraphNode)
        entry_node = entry
        if type(entry) == str:
            entry_node = GraphNode(None, entry)
        exit_node = exit
        if type(exit) == str:
            exit_node = GraphNode(exit, None)

        entry_attributes = {**edge_attributes, **entry_attr}
        exit_attributes = {**edge_attributes, **exit_attr}
        for node in node_list:
            if entry_node:
                self.edge(entry_node.exit, node.entry, **entry_attributes)
            if exit_node:
                self.edge(node.exit, exit.entry, **exit_attributes)

        if exit_node:
            return [exit_node]
        else:
            return node_list
        
    
    def edge(self, tail_name: str, head_name: str, label: str=None, **attrs):
        if tail_name not in PlutoFlowchart.node_followers:
            PlutoFlowchart.node_followers[tail_name] = []
        PlutoFlowchart.node_followers[tail_name].append(GoToStep(head_name, label))

        if label:
            escaped_label = html.escape(word_wrap(label)).replace("\\n", "<BR/>")
            label_box = (
                '<<TABLE border="1" cellborder="0" cellspacing="0" BGCOLOR="white">'
                f'<TR><TD>  {escaped_label}</TD></TR>'
                '</TABLE>>'
            )

        self.g.edge(tail_name, head_name, label and label_box or '', **attrs)
        # self.g.edge(tail_name, head_name, label, **attrs)
        

    def get_as_string(self, node: Union[Tree, Token], default: str=""):
        if isinstance(node, Tree):
            return self.code[node.meta.start_pos: node.meta.end_pos]
        if isinstance(node, Token):
            return self.code[node.start_pos: node.end_pos]
        return default
    
    def visit_nodes(self, nodes: list):
        return [(self.visit(node) if node else None) for node in nodes]
    
    @contextmanager
    def sub_chart(self, root: Tree=None, name: str="", graph_attrs: dict={}, base_graph: graphviz.Digraph=None):
        graph_name = "cluster_"
        if root:
            graph_name += get_unique_name(root)
        if name:
            graph_name += f"_{name}"

        if not base_graph:
            base_graph = self.g

        with base_graph.subgraph(name=graph_name) as sub_graph:
            apply_graph_attr(sub_graph, graph_attrs)
            sub_chart = PlutoFlowchart(sub_graph)
            yield sub_chart

        self.steps = {**self.steps, **sub_chart.steps}

    def visit_or_empty_point(self, nodes: List[Union[Tree, Token]]):
        graph_nodes = filter_list(self.visit_nodes(nodes), GraphNode)
        if not graph_nodes:
            graph_nodes.append(self.node('', name=f"{self.g.name}_EMPTY", **ATTR_POINT))
        return graph_nodes
    

    def nodes_to_list(self, nodes: List[Union[Tree, Token]], header: str):
        list_items = []
        for node in nodes:
            node_str = html.escape(self.get_as_string(node))
            list_items.append(Paragraph(node_str, pstyle))
        if list_items:
            return [
                Paragraph(header, pstyle_bi),
                ListFlowable(list_items, **lstyle_bullet)
            ]


    # ============== TOKEN HANDLERS ============== 

    def procedure_definition(self, tree: Tree):
        procedure_declaration_body = get_child(tree, 'procedure_declaration_body')
        preconditions_body = get_child(tree, 'preconditions_body')
        procedure_main_body = get_child(tree, 'procedure_main_body')
        watchdog_body = get_child(tree, 'watchdog_body')
        confirmation_body = get_child(tree, 'confirmation_body')

        self.step('START of Procedure', node_name='START', **ATTR_TERMINAL)
        self.step_num_lookup['START'] = 'START'
        sub_nodes = self.visit(procedure_main_body)
        self.step('END of Procedure', node_name='END', **ATTR_TERMINAL)
        self.step_num_lookup['END'] = 'END'
        self.edges_sequential(sub_nodes, 'START', 'END')

        return GraphNode('START', 'END')
    

    def log_statement(self, tree: Tree):
        return
        text = ""
        for node in tree.children:
            if isinstance(node, Tree) and node.data == 'expression':
                text += self.get_as_string(node)
            elif isinstance(node, Token) and node.type == 'COMMA':
                text += ', '

        return self.node(f"Log {text}", tree, **ATTR_STEP)
    

    def wait_statement(self, tree: Tree):
        return self.step('Wait', details=self.get_as_string(tree), node=tree, **ATTR_WAIT)
    

    def case_statement(self, tree: Tree):
        expression = get_child(tree, 'expression')

        case_blocks = []
        for child in tree.children:
            if get_type(child) == 'case_tag':
                case_str = self.get_as_string(child)
                if case_str[0] == "=":
                    case_str = case_str[1:]
                case_blocks.append({
                    'case_tag': case_str,
                    'step_statements': []
                })
            elif get_type(child) == 'OTHERWISE':
                case_blocks.append({
                    'case_tag': 'otherwise',
                    'step_statements': []
                })
            elif get_type(child) == 'step_statement':
                print(case_blocks[-1]['case_tag'])
                case_blocks[-1]['step_statements'].append(child)

        label = f"Select {self.get_as_string(expression)}"
        branch_node = self.step(label, node=tree, **ATTR_CASE)
        
        case_exits = []
        for i, case_block in enumerate(case_blocks):
            with self.sub_chart(tree, f'CASE{i}', ATTR_GRAPH_CLEAR) as case_chart:
                case_nodes = case_chart.visit_or_empty_point(case_block['step_statements'])
                case_chart.edges_sequential(case_nodes, 
                    entry=branch_node, 
                    entry_attr={'label': case_block['case_tag'], 'tailport': '_'}
                )
                case_exits.append(case_nodes[-1])

        end_node = self.node("", node=tree, name='END', **ATTR_POINT)
        self.edges_parallel(case_exits, exit=end_node, tailport='s', headport='n', arrowhead='none')

        return GraphNode(branch_node.entry, end_node.exit)
    

    def if_statement(self, tree: Tree):
        # Get if/else block statements
        expression = get_child(tree, 'expression')
        statements = {
            'IF': [],
            'ELSE': []
        }
        current_block = 'IF'
        for node in tree.children:
            if get_type(node) == "ELSE":
                current_block = 'ELSE'
            elif get_type(node) == "step_statement":
                statements[current_block].append(node)

        # branch node
        label = f"Is {self.get_as_string(expression)}?"
        branch_node = self.step(label, node=tree, **ATTR_IF)

        # if block
        with self.sub_chart(tree, 'IF', ATTR_GRAPH_CLEAR) as if_chart:
            if_nodes = if_chart.visit_or_empty_point(statements['IF'])
            if_chart.edges_sequential(if_nodes, 
                entry=branch_node, 
                entry_attr={'label': 'Yes'},
            )

        # else block
        with self.sub_chart(tree, 'ELSE', ATTR_GRAPH_CLEAR) as else_chart:
            else_nodes = else_chart.visit_or_empty_point(statements['ELSE'])
            else_chart.edges_sequential(else_nodes, 
                entry=branch_node, 
                entry_attr={'label': 'No', 'tailport': 'e'},
            )

        # end if node
        end_node = self.node("", node=tree, name='END', **ATTR_POINT)
        self.edges_parallel([if_nodes[-1], else_nodes[-1]], exit=end_node, exit_attr={'arrowhead': 'none'})

        return GraphNode(branch_node.entry, end_node.exit)
    

    def assignment_statement(self, tree: Tree):
        variable_reference: Tree = get_child(tree, 'variable_reference')
        return self.step(f'Assign {self.get_as_string(variable_reference)}', 
                         details=self.get_as_string(tree), node=tree, **ATTR_STEP)

    
    def initiate_and_confirm_activity_statement(self, tree):
        activity_call: Tree = get_child(tree, 'activity_call')
        ACTIVITY_STATEMENT: Token = get_child(tree, 'ACTIVITY_STATEMENT')
        continuation_test = get_child(tree, 'continuation_test')

        label = self.get_as_string(activity_call, "<Unknown Activity>")

        return self.step(label, node=tree, **ATTR_STEP)

    
    def initiate_and_confirm_step_statement(self, tree: Tree):
        STEP_NAME: Token = get_child(tree, 'STEP_NAME')
        step_definition: Tree = get_child(tree, 'step_definition')
        continuation_test = get_child(tree, 'continuation_test')    # TODO!

        step_declaration_body: Tree = get_child(step_definition, 'step_declaration_body')
        preconditions_body: Tree = get_child(step_definition, 'preconditions_body')
        step_main_body: Tree = get_child(step_definition, 'step_main_body')
        watchdog_body: Tree = get_child(step_definition, 'watchdog_body')    # TODO!
        confirmation_body: Tree = get_child(step_definition, 'confirmation_body')    # TODO!
        
        introduction_details = []

        if step_declaration_body:
            declarations_list = self.nodes_to_list(
                step_declaration_body.children, 
                'Declarations:'
            )
            if declarations_list:
                introduction_details.append(declarations_list)

        if preconditions_body:
            preconditions_list = self.nodes_to_list(
                preconditions_body.children, 
                'Preconditions:'
            )
            if preconditions_list:
                introduction_details.append(preconditions_list)

        label = STEP_NAME.value
        with self.sub_chart(tree, graph_attrs=ATTR_GRAPH_STEP) as sub_chart:
            sub_chart.g.attr(label=label)
            intro_node = sub_chart.step(f'Introduction {STEP_NAME}', introduction_details, tree, 'INTRO', **ATTR_STEP)
            sub_nodes = sub_chart.visit_or_empty_point([step_main_body])
            self.edges_sequential(sub_nodes, entry=intro_node)

        return GraphNode(intro_node.entry, sub_nodes[-1].exit)


chart = PlutoFlowchart()
chart.create_flowchart("./test2.pluto", 'output/output')

def get_following_steps(chart: PlutoFlowchart, node: str, label: Optional[str]=None) -> List[GoToStep]:
    following_steps = []
    for node_goto in chart.node_followers.get(node, []):
        goto_label = label or node_goto.label or ''
        if node_goto.node_name in chart.step_num_lookup:
            following_steps.append(GoToStep(node_goto.node_name, goto_label))
        else:
            following_steps += get_following_steps(chart, node_goto.node_name, goto_label)
    return following_steps


for node, step in chart.steps.items():
    print(f"{chart.step_num_lookup[node]} <{node}>")
    for step_goto in get_following_steps(chart, node):
        print(f"-> {chart.step_num_lookup[step_goto.node_name]} {step_goto.label}")






# Define the data for the table

table_data = []

table_style = [
    ('WORDWRAP', (0, 1), (-1, -1), True),  # Word wrapping for cells
    ('FONTNAME', (0, 1), (-1, -1), FONT_TABLE['']),
    ('FONTSIZE', (0, 1), (-1, -1), FONT_TABLE_SIZE),
    # ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    # ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, 1), 'LEFT'),
    
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('FONTNAME', (0, 0), (0, -1), FONT_TABLE['b']),
    ('FONTSIZE', (0, 0), (0, -1), FONT_TABLE_SIZE),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0.2*cm),
    ('TOPPADDING', (0, 0), (-1, -1), 0.2*cm),
    # ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 0.2, colors.black),
    ('BOX', (0, 0), (-1, -1), 1, colors.black),
]


for node, step in chart.steps.items():
    table_data.append([step.number_str, Paragraph(step.header, pstyle_b)])
    col = len(table_data) - 1
    table_style.append(('LINEABOVE', (0, col), (-1, col), 1, colors.black))

    if step.details:
        if type(step.details) == str:
            table_data.append(['', Paragraph(step.details.replace("\n", "<br />"), pstyle)])
        elif isinstance(step.details, Paragraph):
            table_data.append(['', step.details])
        elif type(step.details) == list:
            # if details is a list, no formatting is applied
            for detail in step.details:
                table_data.append(['', detail])

    following_steps = get_following_steps(chart, node)
    if following_steps:
        next_steps_p = [Paragraph('Next step(s):', pstyle_bi)]
        for step_goto in get_following_steps(chart, node):
            step_nr_str = chart.step_num_lookup[step_goto.node_name]
            next_steps_p.append(Paragraph(f'-> <font name={FONT_TABLE["b"]}>{step_nr_str}</font> {step_goto.label}', pstyle_indent))
        table_data.append(['', next_steps_p])

# Create a PDF document
doc = SimpleDocTemplate(
    "output/output.pdf", 
    pagesize=portrait(A4),
    topMargin=1*cm,
    bottomMargin=1*cm,
    leftMargin=1*cm,
    rightMargin=1*cm,
)

drawing = svg2rlg("output/output.svg")
drawing.hAlign = 'CENTER'

# scale to fit the width of an A4 page
max_width = doc.width
scale = min(max_width / drawing.minWidth(), 0.5)
drawing.width, drawing.height = drawing.minWidth() * scale, drawing.height * scale
drawing.scale(scale, scale)

#if you want to see the box around the image
drawing._showBoundary = True
        

# Create a table and define its style
table = Table(table_data, colWidths=[1.5*cm, 16*cm])
table.setStyle(TableStyle(table_style))

# Build the PDF document
chart_bounds = drawing.getBounds()
chart_width = doc.width #drawing.width * 1.5 # chart_bounds[2] - chart_bounds[0]
chart_height = max(drawing.height, doc.height) # chart_bounds[3] - chart_bounds[1]
frame_chart = Frame(doc.leftMargin, doc.bottomMargin, chart_width, chart_height, 0, 0, 0, 0, id='flowchart', showBoundary=1)
frame_normal = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
doc.addPageTemplates([
    PageTemplate(id='flowchart', frames=frame_chart, pagesize=(
        frame_chart.width + doc.leftMargin + doc.rightMargin, 
        frame_chart.height + doc.topMargin + doc.bottomMargin
    )),
    PageTemplate(id='normal', frames=frame_normal)
])

elements = []
elements.append(NextPageTemplate('flowchart'))
elements.append(drawing)
elements.append(NextPageTemplate('normal'))
elements.append(PageBreak())
elements.append(table)
# elements.append(NextPageTemplate('OneCol'))
# elements.append(PageBreak())
# elements.append(Paragraph("Mohammed Shafeeque", style))
# elements = [drawing, table]
doc.build(elements)





# # Create the PDF object, using the response object as its "file."
# pdf = canvas.Canvas("simple.pdf", pagesize=letter)

# # Draw things on the PDF. Here's where the PDF generation happens.
# # See the ReportLab documentation for the full list of functionality.
# pdf.drawString(100, 750, "Welcome to Reportlab!")

# # pdf.drawImage("my_circuit.svg", 15, 720)

# # Close the PDF object cleanly, and we're done.
# pdf.showPage()
# pdf.save()



# d = Drawing()
# d.config(fontsize=10, unit=.5)
# d += flow.Terminal().label('START of Procedure')
# d += flow.Arrow()
# # d += elm.EncircleBox([
# #     flow.Terminal().label('inner1'),
# #     flow.Arrow(),
# #     flow.Terminal().label('inner2')
# # ]).label("Encircled")
# # sub_drawing = flow.Process().fill(color='lightgray').label('test_outer')
# d += (t := flow.Process().label('inner1')).fill(color='white')

# d += elm.EncircleBox([
#     t
# ]).label("Encircled").fill(color='#FF0000AA')

# # sub_drawing.Arrow()
# # sub_drawing.Process().label('inner2')
# d += flow.Arrow()
# d += flow.Terminal().label('END of Procedure')
# d.save('test.svg')




# g = graphviz.Digraph('G', format='svg')


# # Set the graph attributes
# g.attr(rankdir='TB')
# g.attr(overlap='true')
# g.attr(fontname='Arial')
# g.node_attr['fontname']='Arial'
# g.attr(compound='true')

# # Add nodes
# g.node('START', shape='box', style='rounded', label='START of Procedure')

# g.node('S1', label='Step 1', shape='box')

# STEP_STYLE={
#     'shape': 'box',
#     'style': 'filled',
#     'fillcolor': "#00ffff",
#     'color': '#0000ff'
# }

# # Create a subgraph
# with g.subgraph(name='cluster_x') as subgraph:
#     subgraph.attr(style='filled', color='lightgrey')
#     subgraph.node('C1', label='Blab', **STEP_STYLE)
#     subgraph.node('C2', label='Clab', shape='box')
#     subgraph.node('C3', label='Dlab', shape='box')
#     subgraph.edge('C1', 'C2')
#     subgraph.edge('C2', 'C3')
#     subgraph.attr(label='Subgraph Box Label', labeljust='l')

# g.node('S2', label='Step 2', shape='box')
# g.node('END', shape='box', style='rounded', label='END of Procedure')

# g.edge('START', 'S1', label=' ')
# g.edge('S1', 'C1', lhead='cluster_x', label=' ')
# g.edge('C3', 'S2')
# g.edge('S2', 'END', label='<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD BGCOLOR="white" ALIGN="CENTER">Your Label</TD></TR></TABLE>>')

# # Render the graph
# g.render('test')