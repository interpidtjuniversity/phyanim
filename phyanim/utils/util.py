import re

NUMBER_PATTERN = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
def is_numeric(string):
    return bool(re.match(NUMBER_PATTERN, string))