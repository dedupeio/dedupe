
import platform

if platform.system() == 'Darwin':
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    Queue = ctx.Queue
    Process = ctx.Process  # type: ignore
    Pool = ctx.Pool
    SimpleQueue = ctx.SimpleQueue
else:
    from multiprocessing import Process, Pool, Queue, SimpleQueue  # type: ignore # noqa 
