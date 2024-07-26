import matplotlib.pyplot as plt

import time
import sys

def tester():
    for i in range(101):
        time.sleep(0.05)
        sys.stdout("\r%d%%" % i)
        # sys.stdout.flush()
        sys.stdout(f"\rgotten here {i}")
