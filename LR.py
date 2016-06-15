import numpy as np
import scipy 

f=open("datos_clasificacion.dat","r")
f.readline()
xx=f.read()
f.close()
xx=xx.split()
x,y,c=[],[],[]
for i in range(len(xx)/3):
	x.append(float(xx[3*i]))
	y.append(float(xx[3*i+1]))
	c.append(int(xx[3*i+2]))

x1,x2=[],[]
x11,y11,x22,y22=[],[],[],[]
xall=[]
Y=[]
M=[]
#Separando por grupo / Discriminando / That's racist
for i in range(len(c)):
	if c[i]==1:
		x1.append([x[i],y[i]])
		x11.append(x[i])
		y11.append(y[i])
		Y.append(1.0)
		
	if c[i]==2:
		x2.append([x[i],y[i]])
		x22.append(x[i])
		y22.append(y[i])
		Y.append(2.0)
	M.append([1,x[i],y[i]])
	xall.append([x[i],y[i]])
	
MTM=np.dot(np.matrix.transpose(np.matrix(M)),np.matrix(M))
MTMi=np.linalg.inv(MTM)
MTMiM=np.dot(MTMi,np.matrix.transpose(np.matrix(M)))
Y=np.array(Y)
a=np.dot(MTMiM,Y)
a=a.tolist()
print a

b=a[0][1]
cc=a[0][2]
a=a[0][0]
print a,b,cc

def ffa(x,y):
	return a+b*x+cc*y
t1,t2,t3=0,0,0
for i in range(len(c)):
	if ffa(x[i],y[i])>1.5 and c[i]==2:
		t1+=1
	if ffa(x[i],y[i])<=1.5 and c[i]==1:
		t2+=1
print t1,t2,len(c)		

N=5
import matplotlib.pyplot as plt
aa=min(min(l) for l in xall)
ba=max(max(l) for l in xall)
x3=np.linspace(aa,ba,N)
y3=[]
for i in range(len(x3)):
	y3.append(1.0*(1.5-a-b*x3[i])/cc)
plt.plot(x3,y3)
plt.plot(x11,y11,'ro')
plt.plot(x22,y22,'ko')
plt.show()	

