from timeit import default_timer as timer

# to be used as decorator to time funcs when needed
def run_with_timer(func,*args):
    def f(*args, **kwargs):
        s = timer()
        out = func(*args, **kwargs)
        e = timer()
        print(f"{func.__name__}: time == {e-s} seconds")
        return out

    return f
