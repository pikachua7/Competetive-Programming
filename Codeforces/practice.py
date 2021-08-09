n=int(input()) #len of string
s1=input()
for i in range((1<<n)):
    s=''
    for j in range(n):
        if i&(1<<j):
            s+=s1[j]
    print(s)