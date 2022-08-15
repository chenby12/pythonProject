import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


raw_data = pd.read_csv('C:/Users/chenb/Desktop/test data.csv', index_col=0)


def model_preprocessing(test_data):
    raw = test_data.drop(['tR (min)', 'Name', 'CMs'], axis=1)
    scale = StandardScaler()
    result = scale.fit_transform(raw)
    columns_list = raw.columns
    result = pd.DataFrame(result)
    result.columns = list(columns_list)
    result.index = result.index + 1
    result['tR (min)'] = test_data['tR (min)']
    result['Name'] = test_data['Name']
    result['CMs'] = test_data['CMs']
    return result


data = model_preprocessing(raw_data)

train, test = train_test_split(data, test_size=0.25)
train_X = train.drop(['tR (min)', 'Name', 'CMs'], axis=1)
train_Y = train['tR (min)']
test_X = test.drop(['tR (min)', 'Name', 'CMs'], axis=1)
test_Y = test['tR (min)']

KNN = KNeighborsRegressor(algorithm='auto', n_neighbors=10, weights='distance')
NN = MLPRegressor(hidden_layer_sizes=(3000, 1200, 750, 500, 250, 100, 50))


def model_establish(model, weight, x_train, y_train, x_test, y_test):
    m = model.fit(x_train, y_train)
    pre = m.predict(x_test)
    pre = pd.DataFrame(pre).set_index(y_test.index)
    pre_results = pre * weight
    return pre_results


nn = model_establish(NN, 0.7, train_X, train_Y, test_X, test_Y)
knn = model_establish(KNN, 0.3, train_X, train_Y, test_X, test_Y)
prediction = nn + knn

test['predicted tR (min)'] = prediction
print(test)
