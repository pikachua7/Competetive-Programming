n=int(input())
li=list(int(i)for i in input().split())
p=int(input())
dp=[-1]*(n)
s=set()
for i in range(p):
    a,b=map(int,input().split())
    a,b=a-1,b-1
    if dp[a]==-1 and dp[b]==-1:
        s.add(a+1)
        s.add(b+1)
        dp[a] , dp[b] = li[a]+li[b] , li[a]+li[b] 
    elif dp[a]==-1 and dp[b]>=0:
        s.add(a+1)
        dp[a] = max(dp[a],li[a]+li[b])
    elif dp[a]>=0 and dp[b]==-1:
        s.add(b+1)
        dp[b] = max(dp[b],li[a]+li[b])
print(dp,s)



        
    