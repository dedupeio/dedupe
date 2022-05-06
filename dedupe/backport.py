import platform

if platform.system() == "Darwin":
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    Queue = ctx.Queue
    Process = ctx.Process  # type: ignore
    Pool = ctx.Pool
    SimpleQueue = ctx.SimpleQueue
    Lock = ctx.Lock
    RLock = ctx.RLock
else:
    from multiprocessing import Process, Pool, Queue, SimpleQueue, Lock, RLock  # type: ignore # noqa
