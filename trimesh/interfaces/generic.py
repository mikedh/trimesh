import os
import logging
import platform
import subprocess

from string import Template
from tempfile import NamedTemporaryFile
from subprocess import check_call

from .. import exchange

# create a default logger
log = logging.getLogger('trimesh.interfaces')


class MeshScript:

    def __init__(self,
                 meshes,
                 script,
                 tmpfile_ext='stl',
                 debug=False,
                 **kwargs):

        self.debug = debug
        self.kwargs = kwargs
        self.meshes = meshes
        self.script = script
        self.tmpfile_ext = tmpfile_ext

    def __enter__(self):
        # windows has problems with multiple programs using open files so we close
        # them at the end of the enter call, and delete them ourselves at the
        # exit
        self.mesh_pre = [
            NamedTemporaryFile(
                suffix='.{}'.format(
                    self.tmpfile_ext),
                mode='wb',
                delete=False) for i in self.meshes]
        self.mesh_post = NamedTemporaryFile(
            suffix='.{}'.format(
                self.tmpfile_ext),
            mode='rb',
            delete=False)
        self.script_out = NamedTemporaryFile(mode='wb',
                                             delete=False)

        # export the meshes to a temporary STL container
        for mesh, file_obj in zip(self.meshes, self.mesh_pre):
            mesh.export(file_obj.name)

        self.replacement = {'MESH_' + str(i): m.name
                            for i, m in enumerate(self.mesh_pre)}
        self.replacement['MESH_PRE'] = str(
            [i.name for i in self.mesh_pre])
        self.replacement['MESH_POST'] = self.mesh_post.name
        self.replacement['SCRIPT'] = self.script_out.name

        script_text = Template(self.script).substitute(self.replacement)
        if platform.system() == 'Windows':
            script_text = script_text.replace('\\', '\\\\')

        self.script_out.write(script_text.encode('utf-8'))

        # close all temporary files
        self.script_out.close()
        self.mesh_post.close()
        for file_obj in self.mesh_pre:
            file_obj.close()
        return self

    def run(self, command):
        command_run = Template(command).substitute(
            self.replacement).split()
        # run the binary
        # avoid resourcewarnings with null
        with open(os.devnull, 'w') as devnull:
            startupinfo = None
            if platform.system() == 'Windows':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            if self.debug:
                # in debug mode print the output
                stdout = None
            else:
                stdout = devnull

            if self.debug:
                log.info('executing: {}'.format(' '.join(command_run)))
            check_call(command_run,
                       stdout=stdout,
                       stderr=subprocess.STDOUT,
                       startupinfo=startupinfo)

        # bring the binaries result back as a set of Trimesh kwargs
        mesh_results = exchange.load.load_mesh(self.mesh_post.name,
                                               **self.kwargs)

        return mesh_results

    def __exit__(self, *args, **kwargs):
        if self.debug:
            print('MeshScript.debug: not deleting {}'.format(
                self.script_out.name))
            return
        # delete all the temporary files by name
        # they are closed but their names are still available
        os.remove(self.script_out.name)
        for file_obj in self.mesh_pre:
            os.remove(file_obj.name)
        os.remove(self.mesh_post.name)
