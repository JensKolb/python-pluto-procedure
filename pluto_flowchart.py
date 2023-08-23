from contextlib import contextmanager
from typing import Dict, List, NamedTuple, Optional, Type, TypeVar, Union
from pluto_parser.parser import PlutoParser
from lark import Token, Tree
from lark.visitors import Interpreter
import graphviz
import html
import re

from reportlab.platypus import (
    Paragraph, ListFlowable, Flowable
)

from styles import *
from util import *

# NOTE: The naming 'node' is ambiguous here:
# The PlutoFlowchart interpreter visits nodes of the parse tree that was
# created by the lark parser, these will be called 'tree nodes'.
# Based on the tree nodes, the interpreter will create nodes on the 
# flowchart graph, these will be called 'graph nodes'.
# To be able to abstract an entire thread of graph nodes as just one node,
# we will use objects of the class GraphNode, which define 
# an entry and exit graph node.
# This can be confusing and refactoring should be considered here, but the type
# of a so called 'node' can usually be inferred by its python object type:
# (Tree | Token) (alias TreeNode) => tree node
# (str | GraphNode) => graph node


class PlutoFlowchart(Interpreter):
    def __init__(self, code: str, graph: graphviz.Digraph=None, step_store: StepStorage=None):
        super().__init__()

        self.code = code

        self.g = graph
        if not self.g:
            self.g = graphviz.Digraph('G', format='svg')
            apply_graph_attr(self.g, ATTR.GRAPH.BASE)

        self.steps_store = step_store
        if not self.steps_store:
            self.steps_store = StepStorage()

    
    @contextmanager
    def sub_chart(self, root: Tree=None, name: str="", graph_attrs: dict={}, 
                  base_graph: graphviz.Digraph=None):
        """Creates a new sub chart with its own sub graph and provides it as context

        :param root: tree node to get the graphs name from, defaults to None
        :param name: name if no tree_node was supplied, if both were supplied this is a suffix, defaults to ""
        :param graph_attrs: graph attributes, defaults to {}
        :param base_graph: graph to create the sub graph from, defaults to None
        :yield: the sub chart
        """
        graph_name = "cluster_"     # this prefix is required for sub graphs
        if root:
            graph_name += get_unique_name(root)
        if name:
            graph_name += f"_{name}"

        if not base_graph:
            base_graph = self.g

        with base_graph.subgraph(name=graph_name) as sub_graph:
            apply_graph_attr(sub_graph, graph_attrs)
            sub_chart = PlutoFlowchart(self.code, sub_graph, self.steps_store)
            yield sub_chart


    @staticmethod
    def create_flowchart(pluto_file: str, output_file: str) -> 'PlutoFlowchart':
        """Creates a flowchart based on a pluto script

        :param pluto_file: path to a pluto script
        :param output_file: file name of the output SVG file
        """
        code = ""
        with open(pluto_file) as f:
            code = f.read()

        tree = PlutoParser.parse(code)

        chart = PlutoFlowchart(code)
        chart.visit(tree)
        chart.g.render(outfile=f'{output_file}.svg')

        return chart


    def node(self, label: str, tree_node: TreeNode=None, 
                    name: str=None, html_label: bool=False,
                    **node_attributes) -> GraphNode:
        """Creates a node in the flowchart graph

        :param label: label of the node
        :param tree_node: tree node object to get the node name from, defaults to None
        :param name: node name if no tree_node was supplied, defaults to None
        :param html_label: if the label is in HTML format, defaults to False
        :return: GraphNode with both entry and exit set to the name of the new node
        """
        if not name and not tree_node:
            return
        node_name = get_unique_name(tree_node, name) if tree_node else name

        node_label = label

        if not html_label:
            node_label = html.escape(word_wrap(label))

        self.g.node(node_name, label=node_label, **node_attributes)
        
        return GraphNode(node_name, node_name)
    

    def step(self, step: Step, tree_node: TreeNode=None, suffix: str='', 
             show_number: bool=True, **node_attributes) -> GraphNode:
        """Creates a step entry and a corresponding node in the flowchart

        :param step: the step to insert
        :param tree_node: tree node object to create the steps node_name from, defaults to None
        :param suffix: optional suffix to add to the node name, defaults to ''
        :param add_number: if true, adds the step number in the node, defaults to True
        :return: GraphNode with both entry and exit set to the name of the new node
        """
        if not step.node_name:
            step.node_name = get_unique_name(tree_node, suffix)

        self.steps_store.add_step(step) 

        if not step.code_pos and tree_node and tree_node.meta:
            step.code_pos = (tree_node.meta.line, tree_node.meta.column)

        if show_number:
            escaped_label = html.escape(word_wrap(step.header)).replace("\\n", "<BR/>")
            node_label = (
                '<<TABLE border="0" cellborder="0" cellspacing="0" ALIGN="CENTER">'
                f'<TR><TD ALIGN="LEFT" VALIGN="TOP"><I><B>{step.number_str}</B></I>'
                f'</TD><TD ALIGN="CENTER" >{escaped_label}</TD></TR>'
                '</TABLE>>'
            )
        else:
            node_label = step.header

        tooltip = f'Step {step.number_str}'

        return self.node(
            label=node_label, 
            name=step.node_name, 
            html_label=show_number, 
            **{'tooltip': tooltip, **node_attributes}
        )
    

    def edges_sequential(self, graph_nodes: List[GraphNode], entry: Union[str, GraphNode]=None, 
                         exit: Union[str, GraphNode]=None, entry_attr: dict={}, 
                         exit_attr: dict={}, **edge_attributes) -> List[GraphNode]:
        """Creates edges between all supplied graph nodes sequentially

        :param graph_nodes: list of all GraphNodes to connect
        :param entry: entry node, defaults to None
        :param exit: exit node, defaults to None
        :param entry_attr: attribudes for the entry edge, is merged with **edge_attributes
        :param exit_attr: attribudes for the exit edge, is merged with **edge_attributes
        :param **edge_attributes: attributes for the edges
        :return: list of all the nodes that were connected
        """
        node_list = filter_list(graph_nodes, GraphNode)
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
    

    def edges_parallel(self, graph_nodes: List[GraphNode], entry: Union[str, GraphNode]=None, 
                       exit: Union[str, GraphNode]=None, entry_attr: dict={}, 
                       exit_attr: dict={}, **edge_attributes) -> List[GraphNode]:
        """Creates edges from entry and exit to all graph nodes in parallel

        :param graph_nodes: list of all GraphNodes to connect
        :param entry: entry node, defaults to None
        :param exit: exit node, defaults to None
        :param entry_attr: attribudes for the entry edge, is merged with **edge_attributes
        :param exit_attr: attribudes for the exit edge, is merged with **edge_attributes
        :param **edge_attributes: attributes for the edges
        :return: list of all exit nodes
        """
        node_list = filter_list(graph_nodes, GraphNode)
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
        
    
    def edge(self, from_graph_node: str, to_graph_node: str, label: str=None, **attrs):
        """creates an edge between two nodes and stores this connection in node_followers
        """
        self.steps_store.new_goto(from_graph_node, to_graph_node, label)

        if label:
            escaped_label = html.escape(word_wrap(label)).replace("\\n", "<BR/>")
            label_box = (
                '<<TABLE border="1" cellborder="0" cellspacing="0" BGCOLOR="white">'
                f'<TR><TD>  {escaped_label}</TD></TR>'
                '</TABLE>>'
            )

        self.g.edge(from_graph_node, to_graph_node, label and label_box or '', **attrs)
        

    def get_code_snippet(self, tree_node: Optional[TreeNode], default: str=""):
        """Gets the code snippet that this tree_node represents 

        :param tree_node: tree node to get as string
        :param default: return this if tree_node is None, defaults to ""
        :return: code snippet
        """
        if isinstance(tree_node, Tree):
            return self.code[tree_node.meta.start_pos: tree_node.meta.end_pos]
        if isinstance(tree_node, Token):
            return self.code[tree_node.start_pos: tree_node.end_pos]
        return default
    
    
    def visit_nodes(self, tree_nodes: List[TreeNode]):
        """Visits all given tree nodes"""
        return [(self.visit(node) if node else None) for node in tree_nodes]


    def visit_or_empty_point(self, tree_nodes: List[TreeNode]) -> List[GraphNode]:
        """Visits all given tree nodes and returns the resulting graph nodes.
        If there are no resulting graph nodes, an empty graph node is created.

        :param tree_nodes: The tree nodes to visit
        :return: list of resulting graph nodes
        """
        graph_nodes = filter_list(self.visit_nodes(tree_nodes), GraphNode)
        if not graph_nodes:
            graph_nodes.append(self.node('', name=f"{self.g.name}_EMPTY", **ATTR.POINT))
        return graph_nodes
    

    def nodes_to_list(self, tree_nodes: List[TreeNode], header: str) -> List[Flowable]:
        """Assembles the given tree nodes as code snippets in a ListFlowable
        of bullet points for the final document. (Used for step details)

        :param tree_nodes: list of nodes to include
        :param header: title of the list
        :return: list of flowables that make up the final list to display
        """
        list_items = []
        for node in tree_nodes:
            node_str = html.escape(self.get_code_snippet(node))
            list_items.append(Paragraph(node_str, PSTYLE.N))
        return [
            Paragraph(header, PSTYLE.BI),
            ListFlowable(list_items, **LSTYLE.BULLET)
        ]



    # ============== TOKEN HANDLERS ============== 
    # The following functions each handle the tree node of their function name.
    # They should each return a GraphNode to which surrounding nodes 
    # can connect to.

    def procedure_definition(self, tree: Tree):
        procedure_declaration_body = get_child(tree, 'procedure_declaration_body')
        preconditions_body = get_child(tree, 'preconditions_body')
        procedure_main_body = get_child(tree, 'procedure_main_body')
        watchdog_body = get_child(tree, 'watchdog_body')
        confirmation_body = get_child(tree, 'confirmation_body')

        self.step(Step(
                header='START of Procedure', 
                node_name='START', 
                number_str='START',
                code_pos=(tree.meta.line, tree.meta.column)
            ), 
            show_number=False, 
            **ATTR.TERMINAL
        )
        sub_nodes = self.visit(procedure_main_body)
        self.step(Step(
                header='END of Procedure', 
                node_name='END', 
                number_str='END',
                code_pos=(tree.meta.end_line, tree.meta.end_column)
            ), 
            show_number=False, 
            **ATTR.TERMINAL
        )
        self.edges_sequential(sub_nodes, 'START', 'END')

        return GraphNode('START', 'END')
    

    def wait_statement(self, tree: Tree):
        return self.step(Step(
                header='Wait', 
                details=self.get_code_snippet(tree)
            ), 
            tree_node=tree, 
            **ATTR.WAIT
        )
    

    def case_statement(self, tree: Tree):
        expression = get_child(tree, 'expression')

        case_blocks = []
        for child in tree.children:
            if get_type(child) == 'case_tag':
                case_str = self.get_code_snippet(child)
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
                case_blocks[-1]['step_statements'].append(child)

        label = f"Select {self.get_code_snippet(expression)}"
        branch_node = self.step(Step(
                header=label
            ), 
            tree_node=tree, 
            **ATTR.CASE
        )
        
        case_graphs = []
        case_exits = []
        for i, case_block in enumerate(case_blocks):
            with self.sub_chart(tree, f'CASE{i}', ATTR.GRAPH.CLEAR) as case_chart:
                case_nodes = case_chart.visit_or_empty_point(case_block['step_statements'])
                case_chart.edges_sequential(case_nodes, 
                    entry=branch_node, 
                    entry_attr={'label': case_block['case_tag'], 'tailport': '_'}
                )
                case_graphs.append(case_chart.g.name)
                case_exits.append(case_nodes[-1])

        # with self.g.subgraph(name=f'{get_unique_name(tree)}_CASES') as cases_graph:
        #     cases_graph.attr(rankdir='LR', rank='same')
        #     for entry, next_entry in zip(case_graphs, case_graphs[1:]):
        #         cases_graph.edge(entry, next_entry, ltail=self.g.name, lhead=self.g.name, style='invis')

        end_node = self.node("", tree_node=tree, name='END', **ATTR.POINT)
        self.edges_parallel(
            graph_nodes=case_exits, 
            exit=end_node, 
            tailport='s', 
            headport='n', 
            arrowhead='none'
        )

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
        label = f"Is {self.get_code_snippet(expression)}?"
        branch_node = self.step(Step(
                header=label
            ), 
            tree_node=tree, 
            **ATTR.IF
        )

        # if block
        with self.sub_chart(tree, 'IF', ATTR.GRAPH.CLEAR) as if_chart:
            if_nodes = if_chart.visit_or_empty_point(statements['IF'])
            if_chart.edges_sequential(if_nodes, 
                entry=branch_node, 
                entry_attr={'label': 'Yes'},
            )

        # else block
        with self.sub_chart(tree, 'ELSE', ATTR.GRAPH.CLEAR) as else_chart:
            else_nodes = else_chart.visit_or_empty_point(statements['ELSE'])
            else_chart.edges_sequential(else_nodes, 
                entry=branch_node, 
                entry_attr={'label': 'No', 'tailport': 'e'},
            )

        # end if node
        end_node = self.node("", tree_node=tree, name='END', **ATTR.POINT)
        self.edges_parallel(
            graph_nodes=[if_nodes[-1], else_nodes[-1]], 
            exit=end_node, 
            exit_attr={'arrowhead': 'none'}
        )

        return GraphNode(branch_node.entry, end_node.exit)
    

    def assignment_statement(self, tree: Tree):
        variable_reference: Tree = get_child(tree, 'variable_reference')
        return self.step(Step(
                header=f'Assign {self.get_code_snippet(variable_reference)}', 
                details=self.get_code_snippet(tree)
            ), 
            tree_node=tree, 
            **ATTR.STEP
        )

    
    def initiate_and_confirm_activity_statement(self, tree):
        activity_call: Tree = get_child(tree, 'activity_call')
        ACTIVITY_STATEMENT: Token = get_child(tree, 'ACTIVITY_STATEMENT')
        continuation_test = get_child(tree, 'continuation_test')    # TODO!

        label = self.get_code_snippet(activity_call, "<Unknown Activity>")

        return self.step(Step(
                header=label, 
                details=f'Refer by: {ACTIVITY_STATEMENT}' if ACTIVITY_STATEMENT else ''
            ), 
            tree_node=tree, 
            **ATTR.STEP
        )

    
    def initiate_and_confirm_step_statement(self, tree: Tree):
        STEP_NAME: Token = get_child(tree, 'STEP_NAME')
        step_definition: Tree = get_child(tree, 'step_definition')
        continuation_test: Tree = get_child(tree, 'continuation_test')    # TODO!

        step_declaration_body: Tree = get_child(step_definition, 'step_declaration_body')
        preconditions_body: Tree = get_child(step_definition, 'preconditions_body')
        step_main_body: Tree = get_child(step_definition, 'step_main_body')
        watchdog_body: Tree = get_child(step_definition, 'watchdog_body')    # TODO!
        confirmation_body: Tree = get_child(step_definition, 'confirmation_body')    # TODO!
        
        introduction_details = []

        # declarations
        if step_declaration_body:
            declarations_list = self.nodes_to_list(step_declaration_body.children, 'Declarations:')
            introduction_details.append(declarations_list)

        # preconditions
        if preconditions_body:
            preconditions_list = self.nodes_to_list(preconditions_body.children, 'Preconditions:')
            introduction_details.append(preconditions_list)

        label = STEP_NAME.value
        with self.sub_chart(tree, graph_attrs=ATTR.GRAPH.STEP) as sub_chart:
            sub_chart.g.attr(label=label)

            # create indroduction node and adjust numbering for following steps
            intro_node = sub_chart.step(Step(
                    header=f'Introduction {STEP_NAME}', 
                    details=introduction_details
                ), 
                tree_node=tree, 
                suffix='INTRO', 
                **ATTR.STEP
            )
            intro_step_num = self.steps_store.num_lookup[intro_node.entry]
            sub_chart.steps_store = self.steps_store.shallow_copy(f'{intro_step_num}.', 1)

            sub_nodes = sub_chart.visit_or_empty_point([step_main_body])
            self.edges_sequential(sub_nodes, entry=intro_node)

        return GraphNode(intro_node.entry, sub_nodes[-1].exit)