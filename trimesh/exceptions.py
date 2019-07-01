
class ExceptionModule(object):
    """
    Create a dummy module which will raise an exception when attributes
    are accessed.

    For soft dependencies we want to survive failing to import but
    we would like to raise an appropriate error so it is easy to debug.
    """

    def __init__(self, exc):
        self.exc = exc

    def __getattribute__(self, *args, **kwargs):
        raise super(ExceptionModule, self).__getattribute__('exc')


def closure(exc):
    """
    Return a function which will accept any arguments
    but raise the exception when called.

    Parameters
    ------------
    exc : Exception
      Will be raised later

    Returns
    -------------
    failed : function
      When called will raise `exc`
    """
    # scoping will save exception
    def failed(*args, **kwargs):
        raise exc
    return failed
