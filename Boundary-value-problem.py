#Импорт библиотек
import numpy as np
from math import sqrt
#from sympy import integrate
from scipy.integrate import quad

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

#Находим  φ0(x) = a1+a2*x
def φ0(alpha0,alpha1,beta0,beta1,A,B,a,b, x):
    mA = np.zeros((2,2))
    mB = np.zeros(2)
    mA[0,0] = alpha0
    mA[0,1] = alpha0*a+alpha1
    mA[1,0] = beta0
    mA[1,1] = beta0*b+beta1
    mB[0] = A
    mB[1] = B
    #x = seidel(mA,mB,0.0001)
    koef = np.linalg.solve(mA,mB)
    return koef[0]+koef[1]*x

def dφ0(x):
    return 0

def ddφ0(x):
    return 0

#Находим гамма i
def gi(beta0,beta1,a,b,i):
    return -(beta0*(b-a)**2+(i+2)*beta1*(b-a))/(beta0*(b-a)+(i+1)*beta1)
    
#Находим φi(x) = gi(x-a)^(i+1)+(x-a)^(i+2)
def φ1(g,x):
    return g *(x-a)**(1+1)+(x-a)**(1+2)

#Находим φ1'(x)
def dφ1(x):
    return 3*(x**2-2.85714*x+1.85714)

    
#Метод Галеркина
#Вержбицкий Основы численных методов
def Galerkin(alpha0,alpha1,beta0,beta1,A,B,a,b,n):
    h = (b-a)/n
    x=[a+i*h for i in range(n+1)]
    g = gi(beta0,beta1,a,b,1)
    temp1 = φ1(g,b)* dφ1(b) -  φ1(g,a)* dφ1(a)
    temp2 = quad(lambda x: dφ1(x)*dφ1(x),a,b)
    temp3 = quad(lambda x: p(x)*dφ1(x)*φ1(g,x),a,b)
    temp4 = quad(lambda x: q(x)*φ1(g,x)*φ1(g,x),a,b)
    a11 = temp1 - temp2[0] + temp3[0] + temp4[0]
    d1 = quad(lambda x: (f(x)-ddφ0(x)-p(x)*dφ0(x)-q(x)*φ0(alpha0,alpha1,beta0,beta1,A,B,a,b,x))*φ1(g,x),a,b)
    c1 = d1[0]/a11
    Yn = []
    Y_Correct = [y_correct(x[i]) for i in range(len(x))]
    for e in x:
        Yn.append(φ0(alpha0,alpha1,beta0,beta1,A,B,a,b, e) + c1 * φ1(g,e))
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

    print("Краевая задача:")
    print("y''+(2/x)y'-y=0")
    print("y'(1)=0; 1.5y(2)+y'(2)=e^2\n")
    print("Точное решение:")
    print("y(x)=(e^x)/x\n")
    print("\nМетод Галеркина:")
    Answer = Galerkin(alpha0,alpha1,beta0,beta1,A,B,a,b,n)
    print(Answer)
    print("Метод конечных разностей:")
    Answer = BoundaryValueProblem(alpha0,alpha1,beta0,beta1,A,B,a,b,n)
    print(Answer)
    
    

