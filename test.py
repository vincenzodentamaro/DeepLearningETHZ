import pandas as pd

class C(object):
    def __getitem__(self, value):
        print(type(value))
        print(value)
        return value