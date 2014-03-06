from math import sqrt
from math import cos
from math import sin
from math import pi
from copy import deepcopy
def vector_add(a, b, c = 1, d = 1):
    L = []
    for i in range(3):
        L.append(c * a[i] + d * b[i])
    return L

def cross(a, b):
    L = []
    L.append(a[1] * b[2] - a[2] * b[1])
    L.append(a[2] * b[0] - a[0] * b[2])
    L.append(a[0] * b[1] - a[1] * b[0])
    return L

def ic(m, r):
    L = [[0,0,0],
         [0,0,0],
         [0,0,0]]
    s = 0
    for i in range(len(r)):
        s += r[i][1] ** 2 + r[i][2] ** 2
    L[0][0] = m * s
    s = 0
    for i in range(len(r)):
        s -= r[i][0] * r[i][1]
    L[0][1] = m * s
    L[1][0] = m * s
    s = 0
    for i in range(len(r)):
        s -= r[i][0] * r[i][2]
    L[0][2] = m * s
    L[2][0] = m * s
    s = 0
    for i in range(len(r)):
        s += r[i][0] ** 2 + r[i][2] ** 2
    L[1][1] = m * s
    s = 0
    for i in range(len(r)):
        s -= r[i][1] * r[i][2]
    L[1][2] = m * s
    L[2][1] = m * s
    s = 0
    for i in range(len(r)):
        s += r[i][1] ** 2 + r[i][2] ** 2
    L[2][2] = m * s
    return L

def delta(A):
    return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
            A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
            A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

def verse(I):
    L = [[0,0,0],
         [0,0,0],
         [0,0,0]]
    d = delta(I)
    L[0][0] = (I[1][1] * I[2][2] - I[1][2] * I[2][1]) / d
    L[0][1] = (I[1][2] * I[2][0] - I[1][0] * I[2][2]) / d
    L[0][2] = (I[1][0] * I[2][1] - I[1][1] * I[2][0]) / d
    L[1][0] = (I[0][2] * I[2][1] - I[0][1] * I[2][2]) / d
    L[1][1] = (I[0][0] * I[2][2] - I[0][2] * I[2][0]) / d
    L[1][2] = (I[0][1] * I[2][0] - I[0][0] * I[2][1]) / d
    L[2][0] = (I[0][1] * I[1][2] - I[0][2] * I[1][1]) / d
    L[2][1] = (I[0][2] * I[1][0] - I[0][0] * I[1][2]) / d
    L[2][2] = (I[0][0] * I[1][1] - I[0][1] * I[1][0]) / d
    return L

def rotatez(r, theta):
    L = []
    L.append(r[0] * cos(theta) - r[1] * sin(theta))
    L.append(r[0] * sin(theta) + r[1] * cos(theta))
    L.append(r[2])
    return L
    
def mulv(a, L):
    for i in range(len(L)):
        L[i] *= a
    return L

def mulm(a, M):
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] *= a
    return M

def shift(r):
    for i in range(len(r)):
        r[i][0] = r[i][2] * (r[i][2] - 8) / 3
    return r

def mulvm(M, L):
    S = []
    for i in range(len(L)):
        s = 0
        for j in range(len(L)):
            s += L[j] * M[i][j]
        S.append(s)
    return S

def square(u):
    return (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
def f(u1, u2, u01, u02, order, k):
    K = 80
    s = 0
    for i in [0, 1, 2, 3]:
        s += K * (u1[i][order][k] * (1 - sqrt(square(u01[i][order]) / square(u1[i][order]))))
    for i in [0, 1, 2, 3]:
        s -= K * (u2[order][i][k] * (1 - sqrt(square(u02[order][i]) / square(u2[order][i]))))
    return s
def fc(u, u0, F, N = 5):
    for i in range(1, N-1):
        for j in [0, 1, 2, 3]:
            for k in [0, 1, 2]:
                F[i][j][k] = f(u[i-1], u[i], u0[i-1], u0[i], j, k)
def uc(u, rc, rr, N = 5):
    for i in range(N - 1):
        for j in [0, 1, 2, 3]:
            #u[i][j] = vector_add(vector_add(rc[i], rr[i][j]), vector_add(rc[i+1],
                                                                     #rr[i+1][j]),
                                 #d = -1)
             u[i][j] = []
             for k in [0, 1, 2, 3]:
                 u[i][j].append(vector_add(vector_add(rc[i], rr[i][j]), vector_add(rc[i+1], rr[i+1][k]),
                                 d = -1))
def kc(rr, F):
    L = []
    for i in range(len(rr)):
        s = [0,0,0]
        for j in range(4):
            s = vector_add(s, cross(rr[i][j], F[i][j]))
        L.append(s)
    return L
def fcc(F):
    L = []
    for i in range(len(F)):
        s = [0,0,0]
        for j in range(4):
            s = vector_add(s, F[i][j])
        L.append(s)
    return L
def rcc(N = 5):
    L = []
    interval = 8 / (N - 1)
    #print(interval)
    for i in range(N):
        L.append([0, 0, 8 - i * interval])
    return L
def blank(N = 5):
    L = []
    for i in range(N):
        L.append([0, 0, 0])
    return L
def balance(K, Fc, N = 5):
    sign = 1
    for i in range(1, N-1):
        tmp1 = square(K[i])
        tmp2 = square(Fc[i])
        #print(tmp)
        if tmp1 > 0.0016 or tmp2 > 0.0016:
            sign = 0
            break
    return sign

n = int(input('n:'))
THETA = int(input('theta(multiple of 5):'))
radius = 0.4
rc = rcc(n)
rc = shift(rc)
phi = blank(n)
rr = []
for i in range(n):
    temp1 = []
    for j in range(4):
        temp2 = [0, 0, 0]
        if j == 0:
            temp2[0] = -radius
        elif j == 1:
            temp2[1] = radius
        elif j == 2:
            temp2[0] = radius
        elif j == 3:
            temp2[1] = -radius
        temp1.append(temp2)
    rr.append(temp1)
#print('OK')
F = []
for i in range(n):
    temp1 = []
    for j in range(4):
        temp2 = [0.0, 0.0, 0.0]
        temp1.append(temp2)
    F.append(temp1)
Fn = deepcopy(F)
deltaphi = deepcopy(phi)
u = deepcopy(F)
u0 = deepcopy(F)
uc(u0, rc, rr, n)
h = 1e-2; m = 1;
#print('OK!')
for theta in range(2, THETA + 2, 2):
    vc = blank(n)
    Om = blank(n)
    for j in range(4):
        rr[0][j] = rotatez(rr[0][j], pi / 180)
        rr[n-1][j] = rotatez(rr[n-1][j], -pi / 180)
    uc(u, rc, rr, n)
    #print(rc)
    #print(rr)
    fc(u, u0, F, n)
    K = kc(rr, F)
    #print('K:', K)
    #print('rr:', rr)
    #print('u:', u)
    #print('F:', F)
    Kn = kc(rr, Fn)
    Fc = fcc(F)
    #print('OK')

    for num in range(80000000):
        #if num < 20000:
            #h = 0.001
        #else:
            #h = 1e-4
        if balance(K, Fc, n):
            '''print(phi)
            #print(K)
            P = []
            for i in range(n):
                for j in range(4):
                    P.append(point(vector_add(rc[i], rr[i][j])))
                    if j < 3:
                        P.append(line([vector_add(rc[i], rr[i][j]), vector_add(rc[i], rr[i][j+1])]))
                    else:
                        P.append(line([vector_add(rc[i], rr[i][3]), vector_add(rc[i], rr[i][0])]))
                    if i < n - 1 and j == 0:
                        P.append(line([vector_add(rc[i], rr[i][j]), vector_add(rc[i+1], rr[i+1][j])], color = 'red'))
            S = None
            for p in P:
                S += p
            show(S)'''
            print('BALANCE! when theta is', theta, 'num is', num)
            if theta % 10 == 0:
                print('rc:', rc)
                print('rr:', rr)
                print('Fc:', Fc)
                print('K:', K)
            print('\n')
            break
        #else :
            #print('not balance yet!')

        for i in range(1, n-1):
            I = ic(m, rr[i])
            #print(I, rr[i])
            for k in [0, 1, 2]:
                rc[i][k] = rc[i][k] + h * vc[i][k] + h ** 2 / (2 * m) * Fc[i][k]
                tmp = mulvm(verse(I), K[i])[k]
                #print(tmp)
                deltaphi[i][k] = h * Om[i][k] + h ** 2 * tmp
                phi[i][k] = phi[i][k] + deltaphi[i][k]
        for i in range(1, n-1):
            for j in [0, 1, 2, 3]:
                rr[i][j] = vector_add(rr[i][j], cross(deltaphi[i], rr[i][j]))
                rr[i][j] = mulv(radius / sqrt(square(rr[i][j])), rr[i][j])
        uc(u, rc, rr, n)
        fc(u, u0, Fn, n)
        Kn = kc(rr, Fn)
        #print(deltaphi)
        #print(rc)
        #print(u)
        #print('Kn:', Kn)
        #print(Kn)
        #print('\n')
        Fcn = fcc(Fn)
        for i in range(1, n-1):
            I = ic(m, rr[i])
            for k in [0, 1, 2]:
                vc[i][k] = vc[i][k] + h * (Fcn[i][k] + Fc[i][k]) / (2 * m)
                Om[i][k] = Om[i][k] + h / 2 * mulvm(verse(I), vector_add(K[i], Kn[i]))[k]
        F = deepcopy(Fn)
        K = deepcopy(Kn)
        Fc = deepcopy(Fcn)
    else :
        print('Sorry! no balance when theta is', theta)
        print(rc)
        print(rr)
        break
