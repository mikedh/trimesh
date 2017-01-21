import generic as g


class IdentifierTest(g.unittest.TestCase):

    def test_identifier(self):
        count = 100
        for mesh in g.get_meshes(5):
            if not mesh.is_watertight:
                g.log.warning('Mesh %s is not watertight!',
                              mesh.metadata['file_name'])
                continue
            self.assertTrue(mesh.is_watertight)
            g.log.info('Trying hash at %d random transforms', count)
            result = g.deque()
            for i in range(count):
                permutated = mesh.permutate.transform()
                result.append(permutated.identifier)
            ok = (g.np.abs(g.np.diff(result, axis=0)) < 1e-3).all()
            if not ok:
                g.log.error('Hashes on %s differ after transform! diffs:\n %s\n',
                            mesh.metadata['file_name'],
                            str(g.np.diff(result, axis=0)))
            self.assertTrue(ok)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
