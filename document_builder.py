from typing import Any, List, Optional, Tuple, Union
from reportlab.lib import colors

from reportlab.platypus import (
    SimpleDocTemplate, 
    Table, TableStyle, 
    Paragraph, ListFlowable, 
    Flowable, PageTemplate, 
    PageBreak, NextPageTemplate, Frame
)
from reportlab.lib.units import cm
from svglib.svglib import svg2rlg

from pluto_flowchart import PlutoFlowchart
from styles import *
import os

import argparse

parser = argparse.ArgumentParser(description='Creates a flowchart document for a given PLUTO script')
parser.add_argument('pluto_filepath', type=str,
                    help='path to the PLUTO script file')
parser.add_argument('output_name', type=str,
                    help='name or path of the output file')


class PlutoDocument:
    def __init__(self, pluto_filepath: str, output_name: str):
        self.pluto_filepath = pluto_filepath
        self.output_path = output_name
        self.chart: PlutoFlowchart = None
        self.table_data = []
        self.table_style = []

        if (os.path.basename(output_name) == output_name):
            self.output_path = f'./output/{output_name}'

    
    def add_row(self, content: Union[str, Flowable, list], step_num: str='', 
                code_pos: Optional[Tuple[int, int]]=None, row_styles: List[tuple]=[]):
        row = {
            'step_num': step_num,
            'content': None,
            'code_pos': f'{code_pos[0]}' if code_pos else ''
        }

        if type(content) == str:
            row['content'] = Paragraph(content.replace("\n", "<br />"), PSTYLE.N)
        else:
           row['content'] = content

        if not row['content']:
            return 

        self.table_data.append([row['step_num'], row['content'], row['code_pos']])

        if row_styles:
            row = len(self.table_data) - 1
            for style in row_styles:
                self.table_style.append((style[0], (0, row), (-1, row), *style[1:]))
        

    def build_step_table(self) -> Table:
        """Creates the table of steps"""

        self.table_style += TSTYLE.BASE

        steps_store = self.chart.steps_store

        for node, step in steps_store.steps.items():

            self.add_row(Paragraph(step.header, PSTYLE.B), step.number_str, step.code_pos, TSTYLE.STEP)
            # self.table_data.append([step.number_str, Paragraph(step.header, PSTYLE.B)])
            # row = len(self.table_data) - 1
            # self.table_style.append(('LINEABOVE', (0, row), (-1, row), TABLE_BORDER_WIDTH, colors.black))

            if step.details:
                if type(step.details) == list:
                    for detail in step.details:
                        self.add_row(detail)
                else:
                    self.add_row(step.details)


            following_steps = steps_store.get_following_steps(node)
            if following_steps:
                next_steps_p = [Paragraph('Next step(s):', PSTYLE.BI)]
                for step_goto in following_steps:
                    step_nr_str = steps_store.num_lookup[step_goto.to_node]
                    next_steps_p.append(Paragraph(
                        f'-> <font name={FONT_TABLE["b"]}>{step_nr_str}</font> '
                        f'{step_goto.label}', 
                        PSTYLE.INDENT
                    ))
                self.add_row(next_steps_p)
        
        # Create a table and define its style
        table = Table(self.table_data, **TABLE_ATTRIBS)
        table.setStyle(TableStyle(self.table_style))

        return table


    def create(self):

        print("Creating flowchart...")
        self.chart = PlutoFlowchart.create_flowchart(self.pluto_filepath, self.output_path)
        print(f"Flowchart complete, identified {len(self.chart.steps_store.steps)} steps.")

        # Create a PDF document
        print("Creating document...")
        doc_path = f"{self.output_path}.pdf"
        doc = SimpleDocTemplate(
            doc_path,
            **DOC_STYLE
        )

        # convert SVG to drawing
        drawing = svg2rlg(f"{self.output_path}.svg")
        drawing.hAlign = 'CENTER'

        # scale to fit the width of an A4 page
        scale = min(doc.width / drawing.minWidth(), 0.5)
        drawing.width, drawing.height = drawing.minWidth() * scale, drawing.height * scale
        drawing.scale(scale, scale)

        # create template page that grows in length if needed
        chart_width = doc.width
        chart_height = max(drawing.height, doc.height)
        frame_chart = Frame(doc.leftMargin, doc.bottomMargin, chart_width, chart_height, 
                            0, 0, 0, 0, id='flowchart')
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
        elements.append(self.build_step_table())
        doc.build(elements)
        print(f"Document created at {doc_path}.")


if __name__ == "__main__":
    args = parser.parse_args()
    PlutoDocument(args.pluto_filepath, args.output_name).create()