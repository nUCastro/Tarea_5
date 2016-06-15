import numpy as np
import matplotlib.pyplot as plt

def normal(x,mu,sigma):
	x=np.matrix(x)-np.matrix(mu)
	xt=np.matrix.transpose(x)
	constant=(1.0/(2*np.pi*np.linalg.det(sigma)))
	sigmainv=np.linalg.inv(sigma)
	mult=-0.5*(np.dot(x,sigmainv))
	mult=np.dot(mult,xt)
	mult=np.dot(constant,mult)
	return mult

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
X=[]
#Separando por grupo / Discriminando / That's racist
for i in range(len(c)):
	if c[i]==1:
		x1.append([x[i],y[i]])
		x11.append(x[i])
		y11.append(y[i])
		
	if c[i]==2:
		x2.append([x[i],y[i]])
		x22.append(x[i])
		y22.append(y[i])
	X.append([x[i],y[i]])
	
mu1=np.matrix([2,3])
mu2=np.matrix([6,6])
C1=[[5,-2],[-2,5]]
C2=[[1,0],[0,1]]

p1=3.0/8.0
p2=5.0/8.0

def bayes(x,p,mu,sigma):
	f=[]
	for i in range(len(x)):
		tmp=1
		tmp=tmp*p*normal(x[i],mu,sigma)
		f.append(tmp)
	return f
	
tmp1=bayes(X,p1,mu1,C1)
tmp2=bayes(X,p2,mu2,C2)

clase1,clase2=[],[]
for i in range(len(tmp1)):
	clase1.append(1.0*tmp1[i]/(tmp1[i]+tmp2[i]))
	clase2.append(1.0*tmp2[i]/(tmp1[i]+tmp2[i]))
t1,t2=0,0	

for i in range(len(clase1)):
	if clase1[i]>clase2[i] and c[i]==1:
		t1+=1
		print clase1[i],clase2[i],1
	if clase1[i]<clase2[i] and c[i]==2:
		t2+=1
		print clase1[i],clase2[i],2	
print t1,t2	,len(clase1)

	
