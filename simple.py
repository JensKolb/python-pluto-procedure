# This is auto-generated code from the source Pluto file.
from pluto_engine.language import *


class Procedure_(Procedure):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_body.append(stmt_pos_27)
        self.main_body.append(stmt_pos_46)


def stmt_pos_27(caller):
    caller.log(lambda x: "XYX")


def stmt_pos_46(caller):
    caller.log(lambda x: "ABC")
