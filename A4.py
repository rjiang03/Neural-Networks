import numpy as np

#define sigmoid function
def sigmoid(x, weight):
    return 1/(1+np.exp(-np.dot(weight.T, x)))

#get the accuracy of predict data
def accuracy(y, y_hat):
    return 1 - np.sum(np.abs(np.array(y_hat) - np.array(y)))/(2*len(y_hat))

#define a Neural class
class Neural:
    #initialize the class
    def __init__(self, learning_rate = None, max_iteration = None, hidden_layers = None, convergence_threshold = 1e-5):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.hidden_layers = hidden_layers
        self.convergence_threshold = convergence_threshold

    #calculate the multicost based on X and Y, and return cost
    def multicost(self, X, Y):
        l1, l2 = self.feedforward(X)
        inner = np.array(Y).T * (np.log(l2)) - (1 - np.array(Y)).T * (np.log(1 - l2))
        cost = -np.mean(inner)
        return cost

    #get the output of this layer
    def feedforward(self, X):
        l1 = sigmoid(X.T, self.weight1).T
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        l2 = sigmoid(l1.T, self.weight2)
        return l1, l2

    #fit the model
    def fit(self, X, Y):
        N, F = X.shape
        #initialize all the first one is zero
        self.weight1 = np.zeros((F, self.hidden_layers))
        #initialize all the second one as normal distribution
        self.weight2 = np.random.normal(0, 0.1, size = (1 + self.hidden_layers, 3))
        #record the cost of each time
        self.costs = []
        self.costs.append(self.multicost(X, Y))

        #doing Iteration until meet the requirement
        for i in range(self.max_iteration):
            l1, l2 = self.feedforward(X)
            l2_delta = (l2)*(np.array(Y).T - l2)*(1-l2)
            l1_delta = np.dot(l2_delta.T, self.weight2.T) * l1 * (1-l1)

            self.weight2 += np.dot(l1.T, l2_delta.T) * self.learning_rate / N
            self.weight1 += np.dot(X.T, l1_delta)[:, 1:] *self.learning_rate / N

            i += 1
            cost = self.multicost(X, l2.T)
            if np.abs(cost - self.costs[-1]) < self.convergence_threshold and i > 1000:
                print(i)
                break
            self.costs.append(cost)

    def predict(self, X):
        a, b = self.feedforward(X)
        return b


if __name__ == '__main__':
    file = open('A4 - Iris data.txt')
    val_list = file.readlines()
    lists = []
    for string in val_list:
        string = string.split(',')
        lists.append(string)

    lists.pop()

    X = np.array(lists)
    y = X[:, 4]
    X = X[:, 0:4]
    X = X.astype(np.float)

    y = np.where(y == 'Iris-setosa\n', 0, y)
    y = np.where(y == 'Iris-versicolor\n', 1, y)
    y = np.where(y == 'Iris-virginica\n', 2, y)
    Y = y.astype(np.int)

    y = []
    for i in Y:
        a = [0, 0, 0]
        a[i] = 1
        y.append(a)
    Y = np.array(y)

    X_train = np.row_stack((X[:30], X[50:80], X[100:130]))
    Y_train = np.row_stack((Y[:30], Y[50:80], Y[100:130]))

    X_validation = np.row_stack((X[30:40], X[80:90], X[130:140]))
    Y_validation = np.row_stack((Y[30:40], Y[80:90], Y[130:140]))

    X_test = np.row_stack((X[40:50], X[90:100], X[140:150]))
    Y_test = np.row_stack((Y[40:50], Y[90:100], Y[140:150]))

    #set up parameters
    learning_rate = 0.5
    max_iteration = 200000
    hidden_layers = 10
    convergence_threshold = 0.0001

    #start training:
    model = Neural(learning_rate, max_iteration, hidden_layers, convergence_threshold)
    model.fit(X_train, Y_train)

    #get accuracy of training data
    predict_train = model.predict(X_train)
    Y_predict_train = predict_train.T
    Y_predict_tr = Y_predict_train
    for i in range(90):
        Y_predict_tr[i] = np.where(Y_predict_train[i] == np.max(Y_predict_train[i]), 1 , 0)
    print('accuracy on training data is', accuracy(Y_train, Y_predict_tr))

    # get accuracy of validation data
    predict_valid = model.predict(X_validation)
    Y_predict_valid = predict_valid.T
    Y_predict_va = Y_predict_valid
    for i in range(30):
        Y_predict_va[i] = np.where(Y_predict_valid[i] == np.max(Y_predict_valid[i]), 1 , 0)
    print('accuracy on validation data is', accuracy(Y_validation, Y_predict_va))

    #get accuracy of test data
    predict_test = model.predict(X_test)
    Y_predict_test = predict_test.T
    Y_predict_te = Y_predict_test
    for i in range(30):
        Y_predict_te[i] = np.where(Y_predict_test[i] == np.max(Y_predict_test[i]), 1, 0)
    print('accuracy on test data is', accuracy(Y_test, Y_predict_te))

    Y_name= []
    temp3 = Y_predict_test.tolist()
    for label in temp3:
        if label == [1, 0, 0]:
            temp3.append('Iris-setosa')
        elif label == [0,1,0]:
            temp3.append('Iris-versiclor')
        elif label ==[0, 0, 1]:
            temp3.append('Iris-virginica')
    Y_name = temp3[30:]


    print('test prediction is:', Y_name)


    print('pls enter your data, (e.g: 1,1,1,1) :')
    user_input = [float(x) for x in input().split(',')]

    user_input = np.array(user_input)
    print(user_input)
    predict_user = model.predict(np.array([user_input]))
    temp = []

    temp = np.where(predict_user == np.max(predict_user), 1, 0)

    temp3 = temp.T.tolist()
    if temp3[0] == [1, 0, 0]:
        temp3.append('Iris-setosa')
    elif temp3[0] == [0,1,0]:
        temp3.append('Iris-versiclor')
    elif temp3[0] ==[0, 0, 1]:
        temp3.append('Iris-virginica')


    print('Your inputs label prediction is:', temp3[-1])






