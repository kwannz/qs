import queue


class SetQueue(object):
    def __init__(self):
        self.list = list()
        self.qry_queue = queue.Queue()

    def get(self):
        qry_item = self.qry_queue.get()
        self.list.remove(qry_item)
        return qry_item

    def put(self, qry_item):
        if qry_item not in self.list:
            self.list.append(qry_item)
            self.qry_queue.put_nowait(qry_item)
