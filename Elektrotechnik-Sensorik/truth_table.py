
for i in range(8):
    A = i >> 2 & 1
    B = i >> 1 & 1
    C = i & 1

    print(A,B, C)
    print((A|B) & (~A|C)) 