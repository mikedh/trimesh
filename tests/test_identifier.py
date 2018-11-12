try:
    from . import generic as g
except BaseException:
    import generic as g


class IdentifierTest(g.unittest.TestCase):

    def test_identifier(self):
        count = 25
        meshes = g.np.append(list(g.get_meshes(10)),
                             g.get_mesh('fixed_top.ply'))
        for mesh in meshes:
            if not mesh.is_volume:
                g.log.warning('Mesh %s is not watertight!',
                              mesh.metadata['file_name'])
                continue

            g.log.info('Trying hash at %d random transforms', count)
            md5 = g.deque()
            idf = g.deque()
            for i in range(count):
                permutated = mesh.permutate.transform()
                permutated = permutated.permutate.tessellation()

                md5.append(permutated.identifier_md5)
                idf.append(permutated.identifier)

            result = g.np.array(md5)
            ok = (result[0] == result[1:]).all()

            if not ok:
                debug = []
                for a in idf:
                    as_int, exp = g.trimesh.util.sigfig_int(
                        a, g.trimesh.comparison.id_sigfig)

                    debug.append(as_int * (10**exp))
                g.log.error('Hashes on %s differ after transform! diffs:\n %s\n',
                            mesh.metadata['file_name'],
                            str(g.np.array(debug, dtype=g.np.int)))

                raise ValueError('values differ after transform!')

            if md5[-1] == permutated.permutate.noise(
                    mesh.scale / 100.0).identifier_md5:
                raise ValueError('Hashes on %s didn\'t change after noise!',
                                 mesh.metadata['file_name'])

    def test_scene_id(self):
        """
        A scene has a nicely constructed transform tree, so
        make sure transforming meshes around it doesn't change
        the nuts of their identifier hash.
        """
        scenes = [g.get_mesh('cycloidal.3DXML')]

        for s in scenes:
            for geom_name, mesh in s.geometry.items():
                meshes = []
                for node in s.graph.nodes_geometry:
                    T, geo = s.graph[node]
                    if geom_name != geo:
                        continue

                    m = s.geometry[geo].copy()
                    m.apply_transform(T)
                    meshes.append(m)
                if not all(meshes[0].identifier_md5 == i.identifier_md5
                           for i in meshes):
                    raise ValueError(
                        '{} differs after transform!'.format(geom_name))

        assert (scenes[0].geometry['disc_cam_B'].identifier_md5 !=
                scenes[0].geometry['disc_cam_A'].identifier_md5)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
