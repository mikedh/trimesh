try:
    from . import generic as g
except BaseException:
    import generic as g


class ExceptionsTest(g.unittest.TestCase):
    def test_module(self):
        # create an ExceptionWrapper
        try:
            raise ValueError("nah")
        except BaseException as E:
            em = g.trimesh.exceptions.ExceptionWrapper(E)

        # checking isinstance should always return false
        # and NOT raise the error
        assert not isinstance(em, dict)

        try:
            # should re-raise `ValueError('nah')`
            em.hi()
            # if we're here raise an error we don't catch
            raise NameError("should not have worked!!")
        except ValueError:
            # should have re-raised ValueError
            pass


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
