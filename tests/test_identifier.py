import generic as g


class IdentifierTest(g.unittest.TestCase):

    def test_identifier(self):
        count = 25
        for mesh in g.get_meshes(10):
            if not (mesh.is_watertight and
                    mesh.is_winding_consistent):
                g.log.warning('Mesh %s is not watertight!',
                              mesh.metadata['file_name'])
                continue
            self.assertTrue(mesh.is_watertight)
            g.log.info('Trying hash at %d random transforms', count)
            md5 = g.deque()
            idf = g.deque()
            for i in range(count):
                permutated = mesh.permutate.transform()
                permutated = permutated.permutate.tesselation()

                md5.append(permutated.identifier_md5)
                idf.append(permutated.identifier)

            result = g.np.array(md5)
            ok = (result[0] == result[1:]).all()

            if not ok:
                debug = []
                for a in idf:
                    as_int, exp = g.trimesh.util.sigfig_int(a,
                                                            g.trimesh.comparison.identifier_sigfig)

                    debug.append(as_int * (10**exp))
                g.log.error('Hashes on %s differ after transform! diffs:\n %s\n',
                            mesh.metadata['file_name'],
                            str(g.np.array(debug, dtype=g.np.int)))
                self.assertTrue(False)

            if md5[-1] == permutated.permutate.noise(mesh.scale / 100.0).identifier_md5:
                g.log.error('Hashes on %s didn\'t change after noise!',
                            mesh.metadata['file_name'])
                self.assertTrue(False)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
