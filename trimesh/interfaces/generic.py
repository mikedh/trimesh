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
        self.script_out = NamedTemporaryFile(mode='wb', delete=False)
        tempname=self.script_out.name
        
        self.mesh_post  = '%s_out.STL'%tempname

        # export the meshes to a temporary STL container
        self.replacement={}
        self.mesh_pre = []
        for i, m in enumerate(self.meshes):
            f='%s_%d.STL'%(tempname,i+1)
            self.mesh_pre.append(f)
            self.replacement['mesh_%d'%i]=f
            m.export(file_type='stl', file_obj=f)

        self.replacement['mesh_pre']  = str([f for f in self.mesh_pre])
        self.replacement['mesh_post'] = self.mesh_post
        self.replacement['script']    = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        self.script_out.write(script_text.encode('utf-8'))
        self.script_out.close()
        return self

    def run(self, command):
        command_run = Template(command).substitute(self.replacement).split()
        # run the binary
        check_call(command_run)

        # bring the binaries result back as a Trimesh object
        with open(self.mesh_post, mode='rb') as f:
            mesh_result = load_stl(f)
        return mesh_result
    
    def __exit__(self, *args, **kwargs):
        # delete all the freaking temporary files
        from os import remove
        remove(self.script_out.name)
        for f in self.mesh_pre:
            remove(f)
        remove(self.mesh_post)
        pass

