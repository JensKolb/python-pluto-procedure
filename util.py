import re
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Type, TypeVar, Union, Dict
from reportlab.platypus import Paragraph

import graphviz
from lark import Token, Tree

from styles import *

TreeNode = Union[Tree, Token]

class GraphNode(NamedTuple):
    entry: str
    exit: str

@dataclass
class Step:
    header: str = ''
    node_name: str = ''
    number_str: str = ''
    details: Union[str, Paragraph, list] = ''
    table_style: Union[tuple, List[tuple], None] = None
    code_pos: Optional[Tuple[int, int]] = None

@dataclass
class GoTo:
    from_node: str
    to_node: str
    label: str = ''


class StepStorage:
    def __init__(self, prefix: str='', counter: int=0):
        self.prefix = prefix
        self.counter = counter

        self.steps: Dict[str, Step] = {}
        self.num_lookup: Dict[str, str] = {}
        self.node_followers: Dict[str, List[GoTo]] = {}

    def shallow_copy(self, prefix: str='', counter: int=0) -> 'StepStorage':
        copy = StepStorage(prefix, counter)
        copy.steps = self.steps
        copy.num_lookup = self.num_lookup
        copy.node_followers = self.node_followers
        return copy
    
    def add_step(self, step: Step):
        if not step.number_str:
            self.counter += 1
            step.number_str = self.prefix + str(self.counter)

        self.steps[step.node_name] = step
        self.num_lookup[step.node_name] = step.number_str

        return step
    
    def new_goto(self, from_graph_node: str, to_graph_node: str, label: str='') -> GoTo:
        if from_graph_node not in self.node_followers:
            self.node_followers[from_graph_node] = []
        goto = GoTo(from_graph_node, to_graph_node, label)
        self.node_followers[from_graph_node].append(goto)

        return goto
    
    def get_following_steps(self, graph_node: str, label: Optional[str]=None) -> List[GoTo]:
        following_steps = []
        for node_goto in self.node_followers.get(graph_node, []):
            goto_label = label or node_goto.label or ''
            if node_goto.to_node in self.num_lookup:
                following_steps.append(GoTo(graph_node, node_goto.to_node, goto_label))
            else:
                following_steps += self.get_following_steps(node_goto.to_node, goto_label)
        return following_steps
    


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

def get_child(tree: Optional[Tree], name: str) -> Optional[TreeNode]:
    if not tree:
        return None
    for child in tree.children:
        if isinstance(child, Tree) and child.data == name:
            return child
        elif isinstance(child, Token) and child.type == name:
            return child
        
def get_children(tree: Optional[Tree], name: Union[str, List[str]]) -> List[TreeNode]:
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

def get_unique_name(node: TreeNode, suffix: str='') -> str:
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

def get_type(node: Union[TreeNode, str]) -> str:
    if isinstance(node, Tree):
        if isinstance(node.data, Token):
            return node.data.value
        else:
            return node.data
    if isinstance(node, Token):
        return node.type
    if type(node):
        return node

def word_wrap(string: str, max_length: Optional[int]=None) -> str:
    # replace newlines, and multiple whitespaces with single whitespace
    # and remove leading and trailing whitespaces
    if not max_length:
        max_length = LABEL_WRAP_LENGTH
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