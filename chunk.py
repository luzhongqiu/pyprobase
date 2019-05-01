class Chunk:
    """chunk存储"""

    def __init__(self, chunk, chunk_root):
        self.chunk = chunk
        self.chunk_root = chunk_root

    def __eq__(self, other):
        return self.chunk == other.chunk and self.chunk_root == other.chunk_root

    def __hash__(self):
        return hash((self.chunk, self.chunk_root))

    def __str__(self):
        return "<[text]: " + self.chunk + ' [root]: ' + self.chunk_root + '>'

    def __repr__(self):
        return self.__str__()
