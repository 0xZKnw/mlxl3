import time
end = time.monotonic() + 120
value = 1
while time.monotonic() < end:
    for index in range(1000):
        value = (value * 1664525 + index + 1013904223) & 0xffffffff
    time.sleep(0.01)
print(value, flush=True)
