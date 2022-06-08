import platform

if platform.system() == "Darwin":
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    Queue = ctx.Queue
    Process = ctx.Process
    Pool = ctx.Pool
    SimpleQueue = ctx.SimpleQueue
    Lock = ctx.Lock
    RLock = ctx.RLock
else:
    from multiprocessing import (  # type: ignore # noqa
        Lock,
        Pool,
        Process,
        Queue,
        RLock,
        SimpleQueue,
    )
