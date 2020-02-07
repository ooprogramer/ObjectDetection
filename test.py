from multiprocessing import Process, Pipe
aa=99
def f(conn):
    conn.send([42, aa, 'hello'])
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv()[0])   # prints "[42, None, 'hello']"
    p.join()
