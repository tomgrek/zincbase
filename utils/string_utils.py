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
