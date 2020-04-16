#Импорт библиотек
import numpy as np
import math
from scipy.misc import derivative
from scipy.integrate import quad
import matplotlib.pyplot as plt

sqrt = math.sqrt


#Метод конечных разностей
def BoundaryValueProblem(alpha0,alpha1,beta0,beta1,A,B,a,b,n, x):
    h = (b-a)/n
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
    return Yn

#Находим коэффициенты для φ0
def koefφ0(alpha0,alpha1,beta0,beta1,A,B,a,b):
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
    return koef

#Находим  φ0(x) = a1+a2*x
def φ0(koef, x):
    return koef[0]+koef[1]*x

#φ0'(x)
def dφ0(koef, x):
    return derivative(lambda x: φ0(koef, x), x, dx=1e-6)

#φ0''(x)
def ddφ0(koef, x):
    return derivative(lambda x: dφ0(koef, x), x, dx=1e-6)

#Находим гамма i
def gi(beta0,beta1,a,b,i):
    return -(beta0*(b-a)**2+(i+2)*beta1*(b-a))/(beta0*(b-a)+(i+1)*beta1)
    
#Находим φi(x) = gi(x-a)^(i+1)+(x-a)^(i+2)
def φi(g,x,i):
    return g *(x-a)**(i+1)+(x-a)**(i+2)

#Находим φi'(x)
def dφi(g,x,i):
    return  derivative(lambda x: φi(g,x,i), x, dx=1e-6) #3*(x**2-2.85714*x+1.85714)
    
#Метод Галеркина
#Вержбицкий Основы численных методов
def Galerkin(alpha0,alpha1,beta0,beta1,A,B,a,b,n , x):
    h = (b-a)/n
    g = gi(beta0,beta1,a,b,1)
    koef = koefφ0(alpha0,alpha1,beta0,beta1,A,B,a,b)
    
    φ1b = φi(g,b,1)
    dφ1b = dφi(g,b,1)
    φ1a = φi(g,a,1)
    dφ1a = dφi(g,a,1)
    temp1 = φ1b * dφ1b - φ1a * dφ1a
    temp2 = quad(lambda x: dφi(g,x,1)*dφi(g,x,1),a,b)
    temp3 = quad(lambda x: p(x)*dφi(g,x,1)*φi(g,x,1),a,b)
    temp4 = quad(lambda x: q(x)*φi(g,x,1)*φi(g,x,1),a,b)
    a11 = temp1 - temp2[0] + temp3[0] + temp4[0]    
    d1 = quad(lambda x: (f(x)-ddφ0(koef, x)-p(x)*dφ0(koef, x)-q(x)*φ0(koef, x))*φi(g,x,1),a,b)
    c1 = d1[0]/a11
    
    dφ2b = dφi(g,b,2)
    dφ2a = dφi(g,a,2)
    temp1 = φ1b * dφ2b - φ1a * dφ2a
    temp2 = quad(lambda x: dφi(g,x,2)*dφi(g,x,1),a,b)
    temp3 = quad(lambda x: p(x)*dφi(g,x,2)*φi(g,x,1),a,b)
    temp4 = quad(lambda x: q(x)*φi(g,x,2)*φi(g,x,1),a,b)
    a12 = temp1 - temp2[0] + temp3[0] + temp4[0]

    φ2b = φi(g,b,2)
    φ2a = φi(g,a,2)
    temp1 = φ2b * dφ1b - φ2a * dφ1a
    temp2 = quad(lambda x: dφi(g,x,1)*dφi(g,x,2),a,b)
    temp3 = quad(lambda x: p(x)*dφi(g,x,1)*φi(g,x,2),a,b)
    temp4 = quad(lambda x: q(x)*φi(g,x,1)*φi(g,x,2),a,b)
    a21 = temp1 - temp2[0] + temp3[0] + temp4[0]

    temp1 = φ2b * dφ2b - φ2a * dφ2a
    temp2 = quad(lambda x: dφi(g,x,2)*dφi(g,x,2),a,b)
    temp3 = quad(lambda x: p(x)*dφi(g,x,2)*φi(g,x,2),a,b)
    temp4 = quad(lambda x: q(x)*φi(g,x,2)*φi(g,x,2),a,b)
    a22 = temp1 - temp2[0] + temp3[0] + temp4[0]

    d1 = quad(lambda x: (f(x)-ddφ0(koef, x)-p(x)*dφ0(koef, x)-q(x)*φ0(koef, x))*φi(g,x,1),a,b)
    d2 = quad(lambda x: (f(x)-ddφ0(koef, x)-p(x)*dφ0(koef, x)-q(x)*φ0(koef, x))*φi(g,x,2),a,b)

    mA = np.zeros((2,2))
    mB = np.zeros(2)
    mA[0,0] = a11
    mA[0,1] = a12
    mA[1,0] = a21
    mA[1,1] = a22
    mB[0] = d1[0]
    mB[1] = d2[0]
    C = np.linalg.solve(mA,mB)    
    
    Yn = []
    for e in x:
        Yn.append(φ0(koef, e) + C[0]* φi(g,e,1) + C[1]*φi(g,e,2)) 
    return Yn
    
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

def Vyvod(arrX, arrY, corArrY):
    result = "%8s\t%8s\t%12s\t%10s\n" % ('X','Y','Точное решение','Погрешность')
    for i in range(len(arrX)):
        result += "%7f\t%7f\t%7f\t%7f\n" % (arrX[i],arrY[i],corArrY[i],corArrY[i]-arrY[i])
    print(result)


#Описание ОДУ
#y''+(2/x)y'-y=0
def p(x):
    return 0

def q(x):
    return -2*(1+(np.tan(x))**2)

def f(x):
    return 0.0

#Точное решение
def y_correct(x):
    return -np.tan(x)

if __name__ == "__main__":
    #Исходные данные граничных условий
    alpha0 =  1
    alpha1 = 0
    beta0 = 1
    beta1 = 0
    A = 0
    B =  -math.sqrt(3) / 3
    a = 0
    b = math.pi / 6

    n = 10

    x=[a+i*((b-a)/n) for i in range(n+1)]
    Y_Correct = [y_correct(x[i]) for i in range(len(x))]

    print("Краевая задача:")
    print("y''-2*(1+(tg(x))^2)y=0")
    print("y(0)=0; y(pi/6)=-√3/3\n")
    print("Точное решение:")
    print("y(x)=-tg(x)\n")
    print("\nМетод Галеркина:")
    yGalerkin = Galerkin(alpha0,alpha1,beta0,beta1,A,B,a,b,n,x)
    Vyvod(x, yGalerkin, Y_Correct)
    print("Метод конечных разностей:")
    yMKR = BoundaryValueProblem(alpha0,alpha1,beta0,beta1,A,B,a,b,n,x)
    Vyvod(x, yMKR, Y_Correct)

    #Графики
    fig, ax = plt.subplots()
    ax.plot(x, yGalerkin, color='red', label="м. Галеркина")
    ax.plot(x, yMKR, color='blue', label="м. конечных разностей")
    ax.plot(x, Y_Correct, color='black', label="Точное решение")
    ax.legend(loc='upper left')
    plt.show()
    
    
    

