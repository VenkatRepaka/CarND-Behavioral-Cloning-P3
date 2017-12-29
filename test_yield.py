def some_function():
    for x in range(4):
        yield x


for i in some_function():
    print(i)
