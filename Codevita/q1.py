test=int(input())
for _ in range(test):
    numstations=int(input())
    if numstations == 0 :
        print(1)
    elif numstations == 1 :
        print(2)
    elif numstations == 2 :
        print(4)
    else:
        outputli=[0]*(numstations+3)
        outputli[0]=1
        outputli[1]=2
        outputli[2]=4
        for num in range(3,numstations+2):
            outputli[num] = outputli[num-1] + outputli[num-2] + outputli[num-3]
        print(outputli[numstations])
