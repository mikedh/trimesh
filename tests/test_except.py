def test_exception_wrapper():
    # check our exception wrapping behavior
    from trimesh.exceptions import ExceptionWrapper

    try:
        # create a wrapper around a ValueError
        raise ValueError("This should be wrapped!")
    except BaseException as E:
        wrapper = ExceptionWrapper(E)

    # `isinstance` should NOT raise the error that's been wrapped
    assert not isinstance(wrapper, dict)

    # but we should be able to tell that it is wrapping something
    assert isinstance(wrapper, ExceptionWrapper)

    try:
        # check the __getattribute__ path
        wrapper.hi()

        # we wrapped something incorrectly if we got here!
        raise NameError("__getattribute__ should have raised a ValueError!")
    except ValueError:
        # should have re-raised ValueError
        pass

    try:
        # check the __call__ path
        wrapper()

        # we wrapped something incorrectly if we got here!
        raise NameError("__call__ should have raised a ValueError!")

    except ValueError:
        # this is correct
        pass


if __name__ == "__main__":
    test_exception_wrapper()
