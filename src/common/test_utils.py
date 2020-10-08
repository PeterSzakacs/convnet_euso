import random


def random_capitalize(val: str):
    return ''.join(random.choice((str.upper, str.lower))(c) for c in val)
