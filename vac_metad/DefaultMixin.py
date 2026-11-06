class DefaultMixin:
    """This class provides a function that replaces default values of attributes of a child class.

    Note that classes are defined that further inherit from a child of DefaultMixin, and a kwarg.pop/default trick is used to replicate the behaviour set out below.

    Example:
    class Test(DefaultMixin):
        def __init__(self, **kwargs):
            self.x = 1
            self.replace_defaults(kwargs)

    t = Test()       #  keep default value
    t = Test(x=2)    #  change default value
    t = Test(x=2.0)  #  TypeError
    t = Test(y=1)    #  KeyError
    """

    def replace_defaults(self, kwargs):
        if not kwargs:
            return

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"attribute {key} not in {self.__class__.__name__}")

            attr_type = type(getattr(self, key))
            if not isinstance(val, attr_type):
                raise TypeError(
                    f"mismatch for {key}: used {type(val).__name__}, "
                    f"should be {attr_type.__name__}"
                )

            setattr(self, key, val)
