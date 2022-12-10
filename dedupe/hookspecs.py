import pluggy

hookimpl = pluggy.HookimplMarker("dedupe")
hookspec = pluggy.HookspecMarker("dedupe")


@hookspec
def register_variable():
    """Register a variable for use in a datamodel"""
