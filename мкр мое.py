import numpy as np
from math import sqrt


def p(x):
    return 4*x/(2*x+1)

def q(x):
    return -4/(2*x+1)

def f(x):
    return 0.0

def y_correct(x):
    return  3*x+np.exp(-2*x)

def TDMA(a,b,c,f):
    #a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0]*n

    for i in range(n-1):
        alpha.append(-b[i]/(a[i]*alpha[i] + c[i]))
        beta.append((f[i] - a[i]*beta[i])/(a[i]*alpha[i] + c[i]))

    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])

    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]

    return x


alpha0 =  2.0
alpha1 = 1.0
beta0 = 0.0
beta1 = 1.0
A =-9.0
B =  1.0
a = -2.0
b =  0.0

n = 10

h = (b - a)/n

x = np.linspace(a, b, n)
mA = np.zeros((n+1,n+1))
mB = np.zeros(n+1)

aa = np.zeros(n+1)
ac = np.zeros(n+1)
ab = np.zeros(n+1)


for i in range(1,n):
    '''
    ac[i] = 2-q(x[i])*h*h 
    aa[i] = 1-p(x[i])*h/2
    ab[i] = 1+p(x[i])*h/2
    '''
    mB[i]= h*h*f(x[i])
    
    mA[i,i] = 2-q(x[i])*h*h
    mA[i,i+1] = 1+p(x[i])*h/2
    mA[i,i-1] = 1-p(x[i])*h/2
    
'''
aa[0]=2*alpha0*h-3*alpha1
ac[0]=4*alpha1
ab[0]=-alpha1
aa[n-2]=-beta1
ac[n-1]=-4*beta1
ab[n]=2*h*beta0*h+3*beta1
mB[0]=2*A*h
mB[n]= 2*B*h
'''
mA[0,0]= 2*alpha0*h-3*alpha1
mA[0,1]= 4*alpha1
mA[0,2]= -alpha1
mA[n,n-2]= -beta1
mA[n,n-1]= -4*beta1
mA[n,n]= 2*h*beta0*h+3*beta1
mB[0]=2*A*h
mB[n]= 2*B*h
'



#Y = np.linalg.solve(mA,mB)
Y = TDMA(aa,ab,ac,mB)
print(Y)


Y_Correct = []
for i in range(n):
    Y_Correct.append( y_correct(x[i]))

print(Y_Correct)
