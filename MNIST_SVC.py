from sklearn.svm import SVC
from mnist import MNIST


'''Example of format


model = SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
 class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
model.fit(X,Y)
y_hat = model.predict(X_test)
'''


mndataSet = '.\Dataset'
mndata = MNIST()
#mndata.gz =True
mndata = MNIST(mndataSet)
#%%importing data into lists and labels
images,labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()
#%%
c = [.25,.5,1,2,4]
model = SVC(C = 1)
model.fit(images, labels)
y_hat = model.predict(images)

misClass = 0
for x in range(0,45):#len(y_hat)):
    if x%15 == 0:
        print(y_hat[x])
        print(labels[x])
    if y_hat[x] == labels[x]:
        print('ok')
    else:
        misClass = misClass+1

train_error = misClass/45