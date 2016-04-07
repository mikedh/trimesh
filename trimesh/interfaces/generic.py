import numpy as np

from ..io.stl   import load_stl

from string     import Template
from tempfile   import NamedTemporaryFile
from subprocess import check_call

class MeshScript:
    def __init__(self, 
                 meshes, 
                 script):
        self.meshes  = meshes
        self.script  = script

    def __enter__(self):
        self.mesh_pre = [NamedTemporaryFile(suffix='.STL') for i in self.meshes]
        self.mesh_post  = NamedTemporaryFile(suffix='.STL')
        self.script_out = NamedTemporaryFile()

        # export the meshes to a temporary STL container
        for m, f in zip(self.meshes, self.mesh_pre):
            m.export(file_type='stl', file_obj=f.name)

        self.replacement = {'mesh_' + str(i) : m.name for i,m in enumerate(self.mesh_pre)}
        self.replacement['mesh_pre']  = str([i.name for i in self.mesh_pre])
        self.replacement['mesh_post'] = self.mesh_post.name
        self.replacement['script']    = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        self.script_out.write(script_text.encode('utf-8'))
        self.script_out.flush()
        return self

    def run(self, command):
        command_run = Template(command).substitute(self.replacement).split()
        # run the binary
        check_call(command_run)

        # bring the binaries result back as a Trimesh object
        self.mesh_post.seek(0)
        mesh_result = load_stl(self.mesh_post)
        return mesh_result
    
    def __exit__(self, *args, **kwargs):
        # close all the freaking temporary files
        self.mesh_post.close()
        self.script_out.close()
        for f in self.mesh_pre:
            f.close()
