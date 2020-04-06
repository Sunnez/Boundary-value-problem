#Импорт библиотек
import numpy as np
from math import sqrt

#Описание ОДУ
#y''+(2/x)y'-y=0
def p(x):
    return 2/x

def q(x):
    return -1

def f(x):
    return 0.0

#Точное решение
def y_correct(x):
    return np.exp(x)/x

#Метод конечных разностей
def BoundaryValueProblem(alpha0,alpha1,beta0,beta1,A,B,a,b,n):
    h = (b-a)/n
    x=[a+i*h for i in range(n+1)]
    mA = np.zeros((n+1,n+1))
    mB = np.zeros(n+1)
    for i in range(1,n):
        mB[i]= h*h*f(x[i]) 
        mA[i,i] = -(2-q(x[i])*h*h)
        mA[i,i+1] = 1+p(x[i])*h/2
        mA[i,i-1] = 1-p(x[i])*h/2
    mA[0,0]= alpha0*h-alpha1
    mA[0,1]= alpha1
    mA[n,n-1]= -beta1
    mA[n,n]= h*beta0+beta1
    mB[0]=A*h
    mB[n]= B*h
    Yn = seidel(mA,mB,0.0001)
    Y_Correct = [y_correct(x[i]) for i in range(len(x))]
    result = "%8s\t%8s\t%12s\t%10s\n" % ('X','Y','Точное решение','Погрешность')
    for i in range(len(x)):
        result += "%7f\t%7f\t%7f\t%7f\n" % (x[i],Yn[i],Y_Correct[i],Y_Correct[i]-Yn[i])
    return result


#Метод Зейделя-Гаусса
def seidel(A, b, eps):
    n = len(A)
    x = [1.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x

#Служебные функции
def printAns(Y):
    print('_____________________')
    for e in Y:
        print(e,end='\t')
    print()

def printMatr(A):
    for i in range(len(A[0])):
        for j in range(len(A[0])):
                       print("%7f\t" % (A[i,j]),end='')
        print()

if __name__ == "__main__":
    #Исходные данные граничных условий
    alpha0 =  0
    alpha1 = 1.0
    beta0 = 1.5
    beta1 = 1.0
    A = 0
    B =  np.exp(2)
    a = 1
    b = 2
    
    n = 10    
    Answer = BoundaryValueProblem(alpha0,alpha1,beta0,beta1,A,B,a,b,n)
    print(Answer)
    

