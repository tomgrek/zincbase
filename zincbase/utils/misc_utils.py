import math

def chunk(it, max_size):
    """Chunk an iterable into approximately equal size, if it exceeds max_size.

    :returns iterator: Iterator of chunks
    """
    if len(it) < max_size:
        return [it]
    chunks = math.ceil(len(it) / max_size)
    chunksize = math.ceil(len(it) / chunks)
    tots = []
    for i in range(0, len(it), chunksize):
        tots.append(it[i:i+chunksize])
    return tots
