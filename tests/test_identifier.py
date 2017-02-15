import generic as g


class IdentifierTest(g.unittest.TestCase):

    def test_identifier(self):
        count = 50
        for mesh in g.get_meshes(10):
            if not mesh.is_watertight:
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
                debug = [g.np.append(
                    *g.trimesh.util.sigfig_int(a, g.trimesh.comparison.identifier_sigfig)) for a in idf]

                g.log.error('Hashes on %s differ after transform! diffs:\n %s\n',
                            mesh.metadata['file_name'],
                            str(g.np.array(debug)))
            self.assertTrue(ok)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
