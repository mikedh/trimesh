from ..io.stl   import load_stl

from string     import Template
from tempfile   import NamedTemporaryFile
from subprocess import check_call
from os         import remove

class MeshScript:
    def __init__(self, 
                 meshes, 
                 script):
        self.meshes  = meshes
        self.script  = script

    def __enter__(self):
        # windows has problems with multiple programs using open files so we close
        # them at the end of the enter call, and delete them ourselves at the exit
        self.mesh_pre = [NamedTemporaryFile(suffix='.STL', mode='wb', delete=False) for i in self.meshes]
        self.mesh_post  = NamedTemporaryFile(suffix='.STL', mode='rb', delete=False)
        self.script_out = NamedTemporaryFile(mode='wb', delete=False)

        # export the meshes to a temporary STL container
        for mesh, file_obj in zip(self.meshes, self.mesh_pre):
            mesh.export(file_type='stl', file_obj=file_obj.name)

        self.replacement = {'mesh_' + str(i) : m.name for i,m in enumerate(self.mesh_pre)}
        self.replacement['mesh_pre']  = str([i.name for i in self.mesh_pre])
        self.replacement['mesh_post'] = self.mesh_post.name
        self.replacement['script']    = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        self.script_out.write(script_text.encode('utf-8'))
        
        # close all temporary files
        self.script_out.close()
        self.mesh_post.close()
        for file_obj in self.mesh_pre: 
            file_obj.close()

        return self

    def run(self, command):
        command_run = Template(command).substitute(self.replacement).split()
        # run the binary
        check_call(command_run)

        # bring the binaries result back as a Trimesh object
        with open(self.mesh_post.name, mode='rb') as file_obj:
            mesh_result = load_stl(file_obj)
        return mesh_result
    
    def __exit__(self, *args, **kwargs):
        # delete all the temporary files by name
        # they are closed but their names are still available
        remove(self.script_out.name)
        for file_obj in self.mesh_pre:
            remove(file_obj.name)
        remove(self.mesh_post.name)
        

