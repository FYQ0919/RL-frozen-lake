import random

def frozen_lake_map(m,n):
    M = list(range(1,m))
    N = list(range(1,n))
    x=random.sample(M, m//2)
    y=random.sample(N, n//2)
    print(x,y)

    for i in range(m):
        for j in range(n+3):
            if i == 0 and j == 1:
                print('S', end='')
            elif i in x and j in  y:
                print('H', end='')
            elif j == n+2:
                print(',')
            elif i == m-1 and j == n:
                print('G', end="")
            elif j == n+1 or j == 0:
                print('"', end='')
            else:
                print('F', end='')

frozen_lake_map(10,10)



