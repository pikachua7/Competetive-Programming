n=int(input())
li=list(int(i)for i in input().split())
p=int(input())
m=0
li3,li2,li1,sum1,k=[],[-1]*(n),[],[],0
for i in range(p):
    a,b=map(int,input().split())
    a,b=a-1,b-1
    if li2[a]==-1 and li2[b]==-1:
        li3.append([a,b])
        li2[a]=k
        li2[b]=k
        sum1.append(li[a]+li[b])
        li1.append(2)
        m=max(m,sum1[k])
        k=k+1
    elif li2[a]!=-1 and li2[b]==-1:
        li2[b]=li2[a]
        li3[li2[a]].append(b)
        li1[li2[a]]=li1[li2[a]]+1
        sum1[li2[a]]=sum1[li2[a]]+li[b]
        m=max(m,sum1[li2[a]])
    elif li2[b]!=-1 and li2[a]==-1:
        li2[a]=li2[b]
        li3[li2[b]].append(a)
        li1[li2[a]]=li1[li2[a]]+1
        sum1[li2[a]]=sum1[li2[a]]+li[a]
        m=max(m,sum1[li2[a]])
flag=False
for i in range(n):
    if li2[i]==-1:
        if li[i]>=m:
            m=li[i]
            k=i+1
            flag=True
if flag:
    print(k)
else:
    li4,cnt=[],0
    for i in range(k):
        if sum1[i]==m :
            if cnt==0:
                cnt=li1[i]
                li4=li3[i]
            elif li1[i]<cnt:
                cnt=li1[i]
                li4=li3[i]
    li4=sorted([i+1 for i in li4])
    print(*li4)