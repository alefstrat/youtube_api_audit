from datetime import datetime, timedelta
import time
import functools


def schedule(max_iters: int, wait_time: int, time_unit="seconds"):
    # scheduler to automatically pause and restart collections
    # map time_unit to corresponding timedelta argument
    valid_units = {"seconds": "seconds", "minutes": "minutes", "hours": "hours", "days": "days"}

    if time_unit not in valid_units:
        raise ValueError(f"Invalid time unit '{time_unit}'. Choose from: {list(valid_units.keys())}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            num_iters = 1
            while num_iters <= max_iters:
                start = datetime.now()
                next_iter = start + timedelta(**{valid_units[time_unit]: wait_time})

                print(f"\nPerforming iteration number {num_iters}")
                
                func(*args, **kwargs)  # call the wrapped function
                
                num_iters += 1

                if datetime.now() >= next_iter:
                    continue

                else:
                    while datetime.now() < next_iter and num_iters <= max_iters:
                        countdown = next_iter - datetime.now()
                        countdown = str(countdown).split('.')[0]  # remove microseconds
                        print(f"\rNext iteration: Number {num_iters}. Starting in: {countdown}", end='', flush=True)
                        time.sleep(1)

        return wrapper
    return decorator
