import lmdb


class Database:
    def __init__(self, root: str, max_readers: int = 1):
        """Initializes a read-only LMDB environment."""
        self.root = root
        self.env = lmdb.open(
            root,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=max_readers
        )

    def close(self):
        self.env.close()