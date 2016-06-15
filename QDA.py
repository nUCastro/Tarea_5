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
#pi1=3.0/8.0
#pi2=5.0/8.0
pi1,pi2=0.5,0.5

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
	xall.append([x[i],y[i]])
#print xall	

C=np.cov(np.matrix.transpose(np.matrix(xall)))
C1=np.cov(np.matrix.transpose(np.matrix(x1)))
C2=np.cov(np.matrix.transpose(np.matrix(x2)))
C1inv=np.linalg.inv(C1)
C2inv=np.linalg.inv(C2)


mu1=np.mean(x1,axis=0)
mu2=np.mean(x2,axis=0)
print mu1,mu2

Cinv=np.linalg.inv(C)

def func(xi,mui,Cinv,C,pi):
	fi=[]
	xii=[]
	for i in range(len(xi)):
		xii.append([xi[i][0]-mui[0],xi[i][1]-mui[1]])
	for i in range(len(xii)):
		tmp=-0.5*np.log(np.linalg.det(C))
		tmp1=np.dot((np.matrix(xii[i])),Cinv)
		tmp1=np.dot(tmp1,xii[i])
		tmp+=-0.5*tmp1
		tmp+=np.log(pi)
		fi.append(tmp)
	return fi

f1=func(xall,mu1,C1inv,C1,pi1)
f2=func(xall,mu2,C2inv,C2,pi2)

t1,t2,t3=0,0,0
for i in range(len(xall)):
	if f1[i]>f2[i] and c[i]==1:
		#print '1',c[i]
		t1+=1
	if f1[i]<f2[i] and c[i]==2:
		#print '2',c[i]			
		t2+=1
	if f1[i]==f2[i]:
		#print 'lol'
		t3+=1	
print t1,t2,t3,len(x)	

'''
xlen=np.linspace(min(min(l) for l in xall),max(max(l) for l in xall),100)
ylen=np.linspace(min(min(l) for l in xall),max(max(l) for l in xall),100)
xtest=[]
for i in xlen:
	for j in ylen:
		xtest.append([i,j])
f1=func(xtest,mu1,Cinv)
f2=func(xtest,mu2,Cinv)
xp,yp=[],[]
for i in range(len(xtest)):
	if f1[i]>f2[i]:
		xp.append(xtest[i][0])
		yp.append(xtest[i][1])

plt.plot(xp,yp,'ro')
plt.show()	
'''

import matplotlib.pyplot as plt
N=400
a=min(min(l) for l in xall)
b=max(max(l) for l in xall)
x3=np.linspace(a-abs(a/2.0),b+b/2.0,N)
print min(x3),max(x3)
step=(max(x3)-min(x3))/(N*1.0)
y3=[]
for i in range(len(x3)):
	if i%10==0:
		print i
	k=min(x3)
	for j in range(N):
		#Se asume que la Clase 2 esta arriba (y(clase 1) > y(clase 2))
		temp=[[x3[i],k]]
		f1=func(temp,mu1,C1inv,C1,pi1)
		f2=func(temp,mu2,C2inv,C2,pi2)
		if f1>f2:
			k+=step
		if f2>f1:
			k-=step
		#Se espera que despues de N pasos converja		
	y3.append(k)
plt.plot(x3,y3,'-b')
plt.plot(x11,y11,'ro')
plt.plot(x22,y22,'ko')
plt.show()
		
		


