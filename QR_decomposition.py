'''
Created on 28-Feb-2022

@author: ReenaNagrale
'''
'''
Q1.1 

Write a code to generate a random matrix A of size m × n with m > n and calculate its Frobenius norm, ∥ · ∥F . 
The entries of A must be of the form r.dddd (example 5.4316). The inputs are the positive integers m and n 
and the output should display the the dimensions and the calculated norm value.

Q1.2
 Write a code to decide if Gram-Schmidt Algorithm can be applied to columns of a given matrix A through calculation 
 of rank. The code should print appropriate messages indicating whether Gram-Schmidt is applicable on columns 
 of the matrix or not.

Q1.3

Write a code to generate the orthogonal matrix Q from a matrix A by performing the Gram-Schmidt orthogonalization method.
Ensure that A has linearly independent columns by checking the rank. 
Keep generating A until the linear independence is obtained

Q1.4

Write a code to create a QR decomposition of the matrix A by utilizing the code developed in the previous sub-parts 
of this question. Find the matrices Q and R and then display the value ∥A − (Q.R)∥F , where ∥ · ∥F is the Frobenius norm. 
The code should also display the total number of additions, multiplications and divisions to find the result.

'''
import random
import numpy
from math import sqrt
import copy



def Cal_Frobeniusnorm(matrix,rows,cols):
    row = rows
    col = cols
    SumSqrt = 0
    
    for i in range(rows):
        for j in range(cols):
            SumSqrt+=pow(matrix[i][j],2)
    #print(sqrt(SumSqrt))
    return sqrt(SumSqrt)

'''
Function to create random matrix of size m*n 
with elements in format i.dddd

'''

def CreateMatrix(row, col):
    
    row_list=[]
    for i in range(row):
        
        col_list=[]
        for j in range(col):
            col_list.append(random.randrange(10000,59999)/10000)
        row_list.append(col_list)
    matrix_A = row_list
    return matrix_A

'''
Function related to calculate Rank of matrix

'''
#*************************************************************************************#
def swapRows(A,row1,row2):                        #FUNCTION TO SWAP TWO ROWS OF A MATRIX A
    A[row2],A[row1]=A[row1],A[row2]
    return A

def Row_Transformation(A,x,row1,row2):            #FUNCTION TO PERFORM ROW TRANSFORMATION ON ROWS OF A MATRIX
    k=len(A[row2])
    for m in range(k):
        A[row2][m]=A[row2][m] + A[row1][m]*x
    return A

def MatrixRank(A):
    colnum=len(A[0])
    rownum=len(A)
    
    Rank=min(colnum,rownum)                       #RANK IS THE MINIMUM OF colnum AND rownum
    if (rownum>colnum):
        list1=[]
        for i in range(colnum):
            list2=[]
            for j in range(rownum):
                list2.append(A[j][i])
            list1.append(list2)
        list1=list2    
        colnum,rownum=rownum,colnum

    for l in range(Rank):
        if(A[l][l]!=0):
            for n in range(l+1,rownum):
                A=Row_Transformation(A,-(A[n][l]//A[l][l]),l,n)  #INVOKING Row_Transformation FUNCTION
        else:
            flag1=True
            for o in range(l+1,rownum):
                if(A[o][l]!=0):
                    A=swapRows(A,l,o)
                    flag1=False
                    break
            if(flag1):
                for i in range(rownum):
                    A[i][l],A[i][Rank-1]=A[i][Rank-1],A[i][l]
            rownum=rownum-1
        c=0
        for z in A:
            if(z==[0]*colnum):
                c=c+1
    
    return Rank-c

#*************************************************************************************#
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

#Function for subtraction of Matrix

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

#Function of return column of Matrix

def get_acolumn(mat,j):
    return [row[j] for row in mat]

#Function of Generate R from A and Q matrix

def Generate_R(A,Q):
    rows = len(A[0])
    cols = len(Q)
    print("R matrix is of size:",rows,cols)
    R =[]
    for i in range(rows):
        temp = [0]*rows
        for j in range(i,cols):
            
            a = get_acolumn(A,j)
            
            r = sum(x*y for x, y in zip(Q[i],a))
            
            temp[j] = r
            
        R.append(temp)
    
    return R

#Function of decompose Matrix A in form of QR

def Generate_QR(matrix):
    cols = len(matrix[0])
    rows = len(matrix)
    #print(rows,cols)
    
    e=[0]*cols
    r=0
    v=0
    #initialize u[0] = a[0]
    u=[a[0] for a in matrix]
    e[0] = [X/Cal_Frobeniusnorm([u],1,rows) for X in u]
    #R[0][0] = e[0]
    for j in range(1,cols):
        temp = [0]*rows
        a = get_acolumn(matrix,j)
        
        for k in range(0,j):
            
            #v = numpy.dot(a,e[k])
            v = sum(x*y for x, y in zip(e[k],a))
            temp1 = scalarProductMat([e[k]],v)
            temp = [x+y for x, y in zip(temp1[0],temp)]
            #temp = numpy.add(scalarProductMat([e[k]],v),temp)
            
        u[j] = [x-y for x, y in zip(a,temp)]
        #u[j] = numpy.subtract(a,temp) 
        e[j] = [X/Cal_Frobeniusnorm([u[j]],1,rows) for X in u[j]]
    R = Generate_R(matrix,e)
    #transpose to obtain Q
    Q = [[e[j][i] for j in range(len(e))] for i in range(len(e[0]))]
            
    #print("\n Generated Matrix Q:",Q)
    return Q, R

#function to calculate the Operation count for QR decomposition

def total_operation(matrix):
    m = len(matrix) #rows
    n = len(matrix[0]) #columns
    addition_Q = ((m * (n-1) * n)/2) + (m-1)*n
    division_Q = m*n
    multiplication_Q = m*n**2
    total_operation_Q = addition_Q + division_Q + multiplication_Q
    
    addition_R = (n*(n+1)*(m-1))/2
    multiplcation_R = (n*(n+1)*m)/2
    total_operation_R = addition_R + multiplcation_R
    
    total_operation = total_operation_Q + total_operation_R
    print("\nTotal Number of Addition, Division, Multiplication are: {}\n".format(int(total_operation)))
    
                
def main():
    n=input("Enter the number of columns:")
    m = input("Enter number of rows:")
    
    Columns = int(n)
    Rows = int(m)
    
    Amatrix = CreateMatrix(row=int(m), col=int(n))
    print("\n Random matrix created of size: {} * {}".format(m,n))
    print(numpy.matrix(Amatrix))
    l2norm = Cal_Frobeniusnorm(Amatrix, rows=int(m), cols=int(n)) 
    print("\n L2 norm of generated matrix is:", l2norm)
    O_Matrix = copy.deepcopy(Amatrix)
    A_rank = MatrixRank(O_Matrix)
    print("\n Rank of generated matrix is:",A_rank)
    
    if A_rank == Columns and Rows>=Columns:
        print("\n Gram-Schmidt is applicable on columns of the matrix")
        
        Q, R = Generate_QR(O_Matrix)
        print("\n Generated Matrix Q:\n",numpy.matrix(Q))
        print("\n Generated Matrix R:\n",numpy.matrix(R)) 
        
        temp = mat_mul(Q,R)
        result = mat_sub(O_Matrix,temp)
    
        print("\n Frobenius norm of A - QR = ", Cal_Frobeniusnorm(result,Rows,Columns))
        print("\n Round to 4 - Frobenius norm of A - QR = ", round(Cal_Frobeniusnorm(result,Rows,Columns),4))
        total_operation(O_Matrix)
        
    else:
        print("\n Gram-Schmidt is not applicable on the matrix")
    
    
if __name__=="__main__":
    main()