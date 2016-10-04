while True:
    try:
        a =  input().split(" ")
        x, y = a
        sum = 0
        for i in range (int(x)):
            i += 1
            z = (i*(int(y)**i))
            sum = sum+z
        print(sum)
    except EOFError:
        break