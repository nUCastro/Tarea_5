import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

f=open("datos_clasificacion.dat","r")
f.readline()
xx=f.read()
f.close()
xx=xx.split()
x,y,c=[],[],[]
X=[]
for i in range(len(xx)/3):
	x.append(float(xx[3*i]))
	y.append(float(xx[3*i+1]))
	c.append(int(xx[3*i+2]))
	X.append([x[i],y[i]])
FX=X
X=np.matrix(X)
Y=np.array(c)

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression()
print logreg
# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = min(X[:,0]), max(X[:,0])
y_min, y_max = 1.0*x_min, 1.0*x_max
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
print len(X),len(X[i])
U=logreg.predict(FX)
print U
t1,t2=0,0
for i in range(len(U)):
	if U[i]==1 and c[i]==1:
		t1+=1
	if U[i]==2 and c[i]==2:
		t2+=1
print t1,t2		
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
