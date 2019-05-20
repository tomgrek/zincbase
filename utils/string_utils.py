import re

def strip_all_whitespace(line):
    return line.replace(' ', '').strip()

def split_on(line, separator, all=True):
    nesting = 0
    separator_length = len(separator)
    if not line:
        return []
    for i in range(len(line)):
        c = line[i]
        if nesting <= 0 and line[i:i + separator_length] == separator:
            if all:
                return [line[:i]] + split_on(line[i + separator_length:], separator)
            else:
                return [line[:i], line[i + separator_length:]]
        if c in ('[', '('):
            nesting += 1
        elif c in (']', ')'):
            nesting -= 1
            continue
    return [line]

def split_to_parts(line):
    sub_exprs = split_on(line, '(', all=False)
    if len(sub_exprs) != 2:
        raise Exception('Syntax error')
    atoms = split_on(sub_exprs[1][:-1], ',')
    return (atoms[0], sub_exprs[0], atoms[1])

def cleanse(line):
    line = re.sub('[ ./()-]', '_', line)
    return line[0].lower() + line[1:]