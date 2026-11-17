"""Small import/port shim to run the *actual* upstream node code headlessly."""
import importlib.util
import pathlib
import __main__

class _QColor:
    def __init__(self,*a): pass

class _QtGui:
    QColor=_QColor

class BaseNode:
    def __init__(self):
        self.input_data={}
    def pre_step(self):
        self.input_data={port:[] for port in self.inputs}
    def set_input(self,port,value,port_type="signal",coupling=1.0):
        if port in self.inputs:
            self.input_data.setdefault(port,[]).append(float(value)*coupling)
    def get_blended_input(self,port,mode="sum"):
        values=self.input_data.get(port,[])
        return (sum(values) if mode=="sum" else values[0]) if values else None

__main__.BaseNode=BaseNode
__main__.QtGui=_QtGui

def load(path,name):
    path=pathlib.Path(path)
    spec=importlib.util.spec_from_file_location("testnode_"+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module,name)

