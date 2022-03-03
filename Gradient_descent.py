'''
Created on 28-Feb-2022

@author: ReenaNagrale
'''
from _io import open
'''
Consider the last 4 digits of your mobile number (Note : In case there is a 0 in one of the digits replace it by 3). 
Let it be n1n2n3n4. Generate a random matrix A of size n1n2 × n3n4. 
For example, if the last four digits are 2311, generate a random matrix of size 23 × 11. 
Write a code to calculate the l∞ norm of this matrix. Deliverable(s) : The code that generates the results

'''

import random
import numpy
import copy
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

def Cal_L2norm(matrix,rows,cols):
    row = rows
    col = cols
    SumSqrt = 0
    
    for i in range(rows):
        for j in range(cols):
            SumSqrt+=pow(matrix[i][j],2)
    #print(sqrt(SumSqrt))
    return sqrt(SumSqrt)

'''
function to Create Matrix

'''

def CreateMatrix(row, col):
    
    row_list=[]
    for i in range(row):
        
        col_list=[]
        for j in range(col):
            col_list.append(random.randint(1,20))
            #col_list.append(random.randrange(10000,99999)/10000)
        row_list.append(col_list)
    matrix_A = row_list
    return matrix_A

'''
function to calculate Infinite Norm of Matrix

'''
def Cal_InfinteNorm(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    row_sum = [0]*rows
    infnorm = 0
    for i in range(rows):
        temp_sum = 0
        
        for j in range(cols):
            
            temp_sum += abs(matrix[i][j])
        row_sum[i]=temp_sum
    
    infnorm = max(row_sum)
    #print("\n Infinite Norm of Matrix is :",infnorm)
    return infnorm

'''
Function to perform Matrix Transpose

'''
def mat_transpose(a):
    rnum = len(a)
    colnum = len(a[0])
    atranspose = [[a[j][i] for j in range(rnum)] for i in range(colnum)]
    return atranspose
#Function for matrix multiplication


def mat_mul(mat1, mat2):
    
    rows=len(mat1)
    cols = len(mat2[0])
    rows2 = len(mat2)
    result=[]
    #print("Matrix 1:",mat1)
    #print("Matrix 2:",mat2)
    for i in range(rows):
        temp_result=[]
 
    # iterating by column by B
        for j in range(cols):
            temp=0
        # iterating by rows of B
            for k in range(0,rows2):
                #print(i,j,k)
                temp = temp + ( mat1[i][k] * mat2[k][j])
                
            temp_result.append(round(temp,4))
            
        result.append(temp_result)
    
    return result


def mat_sub(mat1,mat2):
    # iterate through rows
    result = [[mat1[i][j] - mat2[i][j]  for j in range(len(mat1[0]))] for i in range(len(mat1))]
    return result


#Function for scalar multiplication of matrix
def scalarProductMat( mat, k):
 
    # scalar element is multiplied
    # by the matrix
    rows = len(mat)
    cols = len(mat[0])
    result = [[0]*cols]*rows
    for i in range( rows):
        for j in range( cols):
            result[i][j] = mat[i][j] * k
    return result


def learning_rate(g,Atransponse,A):
    lrate = 0
    gt = mat_transpose(copy.deepcopy(g))
    numerator = mat_mul(gt,g)  #g transpose * g
    
    temp1 = mat_mul(mat_mul(gt,Atransponse),A) #g transpose * Atransponse * A
    
    denominator = mat_mul(temp1,g)
    lrate = numerator[0][0]/denominator[0][0]
    return lrate


def gradient_descent(previous_x, lrate, g, ATranspose, A,b):
    
    g_curr = mat_sub(mat_mul(mat_mul(ATranspose,A),previous_x), mat_mul(ATranspose,b)) 
    
    temp = [scalarProductMat([g_curr[i]],lrate)[0] for i in range(len(g_curr))]
    #print(temp)
    curr_x = mat_sub(previous_x, temp)
    #print(curr_x)
    return curr_x

def fx_calculation(A_Mat,x,b_mat):
    
    temp=mat_sub(mat_mul(A_Mat,x),b_mat)
    
    temp = Cal_L2norm(matrix=temp, rows=len(temp), cols=len(temp[0]))
    
    fx_value = 0.5 * pow(temp,2)
    
    return fx_value

def createplot(fx):
    
    steps = 0
    fx_line = [i for i in fx if i>0]
    steps = list(range(len(fx_line)))
    plt.style.use('ggplot')
    plt.figure(figsize = (20,8))
    
    sns.lineplot(steps,fx_line)
    plt.xlabel(xlabel='Steps')
    plt.ylabel(ylabel='Cost')
    plt.title("Gradient Descent Cost Function")
    plt.show()
    


def main():
    num = input("Enter last four digit of your number: ")
    
    if int(num)< 9999 and int(num)> 1000:
        n1n2 = int(str(num)[:2])
        n3n4 = int(str(num)[2:4])
        matrix = CreateMatrix(row=n1n2, col=n3n4)
        fx_list=[]
        x_list=[]
        #print("Learning Rate:",lrate, file=outfile)
        
        print("Output is generated in file - Gradient_Descent_Output.txt")
        with open("Gradient_Descent_Output.txt",'w') as outfile:
        
            #print("\n Matrix Created with "+n1n2+" rows and "" columns")
            print("\n Matrix Created with {} rows and {} columns".format(n1n2,n3n4))
            print("\n", numpy.matrix(matrix))
            InfiteNorm = Cal_InfinteNorm(matrix)
            print("\n Infinite Norm of matrix = ",InfiteNorm)
            b = CreateMatrix(n1n2,1)
            print("\n Generated b matrix of size {} * 1".format(n1n2))
            print("\n",numpy.matrix(b))
            previous_x = CreateMatrix(row=n3n4, col=1)
            current_x = CreateMatrix(row=n3n4, col=1)
            i=1
            GTranspose = mat_transpose(matrix)
            print("\n L2 norm of ||x(0) - x(-1)|| = ", Cal_L2norm(mat_sub(current_x,previous_x),n3n4,1))
            x_list.append(current_x)
            
            
            while(Cal_L2norm(mat_sub(previous_x,current_x),n3n4,1)>pow(10,-4)):
                #print(current_x)
            
                g = mat_sub(mat_mul(mat_mul(GTranspose,matrix),current_x),mat_mul(GTranspose,b))
                
                lrate = learning_rate(g,GTranspose,matrix)
               
                previous_x = copy.deepcopy(current_x)
                
                current_x = gradient_descent(current_x, lrate, g, GTranspose, matrix, b)
                
                print("\n Iteration {} and x({}) : \n {} ".format(i,i,numpy.matrix(current_x)))
                
                fx = fx_calculation(matrix,current_x,b)
                print("\n f(x) value for generated x: ",fx)
                
                x_list.append(current_x)
                fx_list.append(round(fx,4))
                
                print("\n L2 norm of ||x({}) - x({}-1)|| = {} ".format(i,i,Cal_L2norm(mat_sub(current_x,previous_x),n3n4,1)))
                #print("\n L2 norm of ||x({}) - x({}-1)|| = {} ".format(i,i,Cal_L2norm(mat_sub(current_x,previous_x),n3n4,1)))
                
                i+=1
               
        createplot(fx_list)
    else:
        print("\n Number is not four digit!!!")
        
        
if __name__=="__main__":
    main()