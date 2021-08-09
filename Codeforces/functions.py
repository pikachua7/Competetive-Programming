# from __future__ import division, print_function
    # from sys import stdin, stdout
    # import bisect
    # import math
    # import heapq
    # i_m = 9223372036854775807

    #1.MATRIX INPUT
    # def matrix(n):
    #     #matrix input
    #     return [list(map(int, input().split()))for i in range(n)]

    # ################################################

    #2.STRING TO LIST
    # def string2intlist(s):
    #     return list(map(int, s))

    #3.MULTIPLES
    # def calculate_sum(a, N):  # sum of a to N
    #     # Number of multiples
    #     m = N / a
    #     # sum of first m natural numbers
    #     sum = m * (m + 1) / 2
    #     # sum of multiples
    #     ans = a * sum
    #     return ans

    #4.SERIES
    # def series(N):
    #     return (N*(N+1))//2

    #5.COUNT 2D MATRIX
    # def count2Dmatrix(i, list):
    #     return sum(c.count(i) for c in list)

    #6.INVERSE MOD
    # def modinv(n, p):
    #     return pow(n, p - 2, p)

    #7.COMBINATIONS (nCr)
    # def nCr(n, r):
    #     i = 1
    #     while i < r:
    #         n *= (n - i)
    #         i += 1
    #     return n // math.factorial(r)

    #8.GCD
    # def GCD(x, y):
    #     x = abs(x)
    #     y = abs(y)
    #     if(min(x, y) == 0):
    #         return max(x, y)
    #     while(y):
    #         x, y = y, x % y
    #     return x

    #9.LCM
    # def LCM(x, y):
    #     return (x * y) // GCD(x, y)


    #10.no.OF DIVISIORS
    # def Divisors(n):
    #     l = []
    #     for i in range(1, int(math.sqrt(n) + 1)):
    #         if (n % i == 0):
    #             if (n // i == i):
    #                 l.append(i)
    #             else:
    #                 l.append(i)
    #                 l.append(n//i)
    #     return l


    #11.IS PRIME
    # def isprime(n):
    #     for i in range(2, int(math.sqrt(n))+1):
    #         if n % i == 0:
    #             return False
    #     return True


    #12.SIEVE
    # prime = []
    # def SieveOfEratosthenes(n):
    #     global prime
    #     prime = [True for i in range(n+1)]
    #     p = 2
    #     while (p * p <= n):
    #         if (prime[p] == True):
    #             for i in range(p * p, n+1, p):
    #                 prime[i] = False
    #         p += 1
    #     f = []
    #     for p in range(2, n):
    #         if prime[p]:
    #             f.append(p)
    #     return f


    #13.DFS
    # q = []
    # def dfs(n, d, v, c):
    #     global q
    #     v[n] = 1
    #     x = d[n]
    #     q.append(n)
    #     j = c
    #     for i in x:
    #         if i not in v:
    #             f = dfs(i, d, v, c+1)
    #             j = max(j, f)
    #             # print(f)
    #     return j
    # # d = {}

    #14.KNAPSACK
    # def knapSack(W, wt, val, n):
    #     K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    #     for i in range(n + 1):
    #         for w in range(W + 1):
    #             if i == 0 or w == 0:
    #                 K[i][w] = 0
    #             elif wt[i-1] <= w:
    #                 K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
    #             else:
    #                 K[i][w] = K[i-1][w]

    #     return K[n][W]

    #15.MODULAR EXPO
    # def modularExponentiation(x, n):
    #     M = 10**9+7
    #     if(n == 0):
    #         return 1
    #     elif (n % 2 == 0):  # n is even
    #         return modularExponentiation((x*x) % M, n//2)
    #     else:  # n is odd
    #         return (x * modularExponentiation((x * x) % M, (n - 1) // 2)) % M

    #16.MOD INVERSE
    # def modInverse(a, m):
    #     m0 = m
    #     y = 0
    #     x = 1

    #     if (m == 1):
    #         return 0

    #     while (a > 1):

    #         # q is quotient
    #         q = a // m

    #         t = m

    #         # m is remainder now, process
    #         # same as Euclid's algo
    #         m = a % m
    #         a = t
    #         t = y

    #         # Update x and y
    #         y = x - q * y
    #         x = t

    #     # Make x positive
    #     if (x < 0):
    #         x = x + m0

    #     return x


    #17.Sum of Digits

    # def getSum(n):

    # sum = 0
    # while (n != 0):

    #     sum = sum + (n % 10)
    #     n = n//10

    # return sum

    #18.MAX While Taking INPUT
    # m=0
    # li=[]
    # for i in input().split():
    #     m=max(m,int(i))
    #     li.append(int(i))
    # print(m)
    # print(li)

    #POWER SET (GENERATE ALL SUBSET/SUBSTRINGS)
    # n=int(input())
    # s1=input()
    # for i in range((1<<n)):
    #     s=''
    #     for j in range(n):
    #         if i&(1<<j):
    #             s+=s1[j]
    #     print(s)



#a<<b = a*(2^b)
#a>>b = a//(2^b)


#User defined
import sys
import os
import math
import copy
from bisect import bisect
from io import BytesIO, IOBase
from math import sqrt,floor,factorial,gcd,log,ceil
from collections import deque,Counter,defaultdict
from itertools import permutations,combinations,accumulate

def Int():               return int(sys.stdin.readline())
def Mint():              return map(int,sys.stdin.readline().split())
def Lstr():              return list(sys.stdin.readline().strip())
def Str():               return sys.stdin.readline().strip()
def Mstr():              return map(str,sys.stdin.readline().strip().split())
def List():              return list(map(int,sys.stdin.readline().split()))
def Hash():              return dict()
def Mod():               return 1000000007
def Mat2x2(n):           return [List() for _ in range(n)]
def Lcm(x,y):            return (x*y)//gcd(x,y)
def dtob(n):             return bin(n).replace("0b","")
def btod(n):             return int(n,2)
def watch(x):            return print(x)
def common(l1, l2):      return set(l1).intersection(l2)
def Most_frequent(list): return max(set(list), key = list.count)


#Matrix input
table = [[0 for x in range(m)] for x in range(n+1)]




#BFS
def bfs(n,adj):
    bfs=[]  # it will keep track of all the nodes visited
    vis=[False]*(n+1) # bool array to keep track of visited nodes
    cnt=0 #to count the no. of components of a graph

    for i in range(1,n+1):  #the driver for loop
        if vis[i]==False:
            cnt+=1
            queue = []  # fifo structure used , queue
            queue.append(i)
            vis[i]=True

            while(len(queue)!=0):
                node=queue.pop(0) # storing the top element and removing it from queue
                bfs.append(node)

                for i in adj[node]: #checking in the top most element adjacent elements
                    if vis[i]==False:
                        vis[i]=True
                        queue.append(i)
    return bfs


#for input of undirected graph
n,m=map(int,input().split())
adj=[[] for i in range(n+1)]
for _ in range(m):
    u,v=map(int,input().split())
    adj[u].append(v)
    adj[v].append(u)

x=bfs(n,adj)
print(x)




#DFS
def dfs(node,vis,adj,dfsarr):
    dfsarr.append(node) #appending the current node
    vis[node]=True #marking it as visited
    for i in adj[node]:
        if vis[i]==False:
            dfs(i,vis,adj,dfsarr) #recursive calling of the nodes

def dfsofgraph(n,adj):
    vis = [False] * (n + 1) #visited array
    dfsarr=[] #path of dfs traversal
    for i in range(1,n+1):#driver for loop
        if vis[i]==False:
            dfs(i,vis,adj,dfsarr) #calling the dfs function
    return dfsarr

n,m=map(int,input().split())
adj=[[] for i in range(n+1)]
for _ in range(m):
    u,v=map(int,input().split())
    adj[u].append(v)
    adj[v].append(u)
x=dfsofgraph(n,adj)
print(x)

#Check cycle in grpah using BFS
def isCycle(n,adj):
    vis=[0]*(n+1)
    for i in range(1,n+1): #driver for loop
        if(vis[i]==0):
            if checkCycle(i,n,adj,vis):
                return True
    return False

def checkCycle(i,n,adj,vis):
    queue=[] #since using bfs we will use queue
    vis[i]=True
    queue.append([i,-1]) #we are appending [node,parent] since it is first node so [node,-1]

    while(len(queue)!=0):
        node=queue[0][0] #current node
        parent=queue[0][1] #parent node
        queue.pop(0) #since we are using queue it is FIFO so removing the element at 0th index

        for i in adj[node]: #finding in adjancey matrix
            if vis[i]==0:
                vis[i]=True
                queue.append([i,node])
            elif parent!=i: #if parent==i means that same parent eleemnt was the previous element of the current node so they cannot form cycle but if they are different means that it has traversed some or the other time and hence cycle is there
                return True
    return False
n, m = map(int, input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

x=isCycle(n,adj)
print(x)


#BIPARTITE GRAPH USING BFS
def checkpartite(adj,n):
    color=[-1]*(n+1) #color array
    for i in range(1,n+1):
        if color[i]==-1:
            if bfsCheck(i,adj,color)==False:
                return False
    return True

def bfsCheck(i,adj,color):
    queue=[]
    queue.append(i)
    color[i]=1
    while(len(queue)!=0):
        node=queue[0]
        queue.pop(0)
        for i in adj[node]:
            if color[i]==-1:
                color[i]=1-color[node] #giving opposite color to the next node                queue.append(i)
            elif color[i]==color[node]:
                return False
    return True

n, m = map(int, input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

if checkpartite(adj,n):
    print('YES BIPARTITE')
else:
    print('NOT BIPARTITE')

#TOPOLOGICAL SORT
def topoSort(n,adj):
    queue,topo=[],[]
    indegree=[0]*(n)
    for i in range(n):
        for j in adj[i]:
            indegree[j]+=1
    for i in range(n):
        if indegree[i]==0:
            queue.append(i)
    while(len(queue)!=0):
        node=queue[0]
        queue.pop(0)
        topo.append(node)
        for i in adj[node]:
            indegree[i]-=1
            if indegree[i]==0:
                queue.append(i)
    return topo

n, m = map(int, input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
x = topoSort(n, adj)
print(x)

#CHECKING CYCLE IN DIRECTED GRAPH USING BFS
def topoSort(n,adj):
    queue,topo=[],[]
    indegree=[0]*(n)
    for i in range(n):
        for j in adj[i]:
            indegree[j]+=1
    for i in range(n):
        if indegree[i]==0:
            queue.append(i)
    cnt=0
    while(len(queue)!=0):
        node=queue[0]
        queue.pop(0)
        cnt+=1
        topo.append(node)
        for i in adj[node]:
            indegree[i]-=1
            if indegree[i]==0:
                queue.append(i)
    if cnt==n:
        return False
    else:
        return True

n, m = map(int, input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
x = topoSort(n, adj)
print(x)


#SHORTEST PATH FROM A NODE TO ALL OTHER NODE USING BFS
def shortestpath(n,adj,src):
    dist=[9999999]*(n+1)
    queue=[]
    queue.append(src)
    dist[src]=0
    #bfs
    while(len(queue)!=0):
        node=queue[0]
        queue.pop(0)
        for i in adj[node]:
            if dist[node]+1 < dist[i]:
                dist[i]=dist[node]+1
                queue.append(i)
    return dist

n, m ,src = map(int, input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

x=shortestpath(n,adj,src)
print(x)

#DIJIKSTRAS ALGO
import heapq
def Dijikstra(n,adj,src):
    dist = [9999999] * (n + 1)
    dist[src]=0
    pq=[[0,src]]
    while len(pq)!=0:
        current_dist,current_node=heapq.heappop(pq)
        for i in adj[current_node]:
            dis=current_dist + i[1]
            if dis < dist[i[0]]:
                dist[i[0]]=dis
                heapq.heappush(pq,[dis,i[0]])
    return dist

n,m,src=map(int,input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u,v,w=map(int,input().split())
    adj[u].append([v,w])
    adj[v].append([u,w])


x=Dijikstra(n,adj,src)
print(x)


#Prims
def Prims(n,adj):
    key=[9999999]*(n)
    mst=[False]*(n)
    parent=[-1]*(n)
    key[0]=0
    parent[0]=-1
    #n-1 edges are only possible
    for i in range(n-1):
        min1=(9999999)
        u=0
        for j in range(n):
             if mst[j]==False and key[j]<min1:
                min1=key[j]
                u=j
        mst[u]=True

        for k in adj[u]:
            node=k[0]
            weight=k[1]
            if mst[node]==False and weight<key[node]:
                parent[node]=u
                key[node]=weight

    return key


n,m=map(int,input().split())
adj = [[] for i in range(n + 1)]
for _ in range(m):
    u,v,w=map(int,input().split())
    adj[u].append([v,w])
    adj[v].append([u,w])
x=Prims(n,adj)
print(x)