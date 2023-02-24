def proc(n):
    if n < 0:
        print('-', end='')
        n = -n
    if n // 10:
        proc(n // 10)
    print(n % 10, end='')


proc(-345)