import numpy as np
# # seed = 111
# # x = np.random.default_rng(seed)
# # print(x)



# a = np.arange(9).reshape(3,3)
# flatten = a.flat[::a.shape[1]+1]
# print(flatten)


# M = np.zeros(9).reshape(3,3)
# diag_values = np.array([0.00001, 0.00001, 1.0])
# np.fill_diagonal(M, diag_values)

# print(M)
k=3
preds = []
X_train = np.array([[1,9,3],[18,7,5],[6,7,8]])
X_test = np.array([[10,11,12],[13,14,15],[16,17,18]])
t_train = np.array([[15,30,45],[60,75,90],[105,120,135]])

for i in range(np.size(X_test, 0)):
    distance_arr = np.empty(np.size(X_train,0))
    for j in range(np.size(X_train, 0)):
        distance = np.linalg.norm(X_test[i] - X_train[j])
        print(distance_arr)
        print(distance)
        distance_arr[j] = distance

        sorted_distances = distance_arr.argsort()
        print(sorted_distances)
        pred_value = t_train[sorted_distances[0:k]]

        prediction_calc = np.sum(pred_value) / k
        preds.append(prediction_calc)

    