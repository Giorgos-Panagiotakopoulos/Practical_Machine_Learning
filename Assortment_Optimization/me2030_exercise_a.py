import os

import matplotlib
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

inputFile = input('Complete path and filename to import: ')
datafolder = os.path.dirname(inputFile)
# Διαβάζει το csv αρχείο και δημιουργεί NumPy array
data = np.genfromtxt(inputFile)
print(data.shape)
# Υπολογισμός των κωδικών προϊόντων που περιέχεται στο αρχείο
# Ο αριθμός προκύπτει από πλήθος των στηλών διαιρώντας με το 4,
# μιας και για κάθε κωδικό προϊόντος το αρχείο περιέχει 4 στήλες
# με πληοροφορίες γι' αυτό.
ProductsPerAssortment = int((data.shape[1] - 1) / 4)
# Το πρώτο στοιχείο του tuple που επιστρέφει η μέθοσος shape μας δίνει το πλήθος των προϊόντων
howManyAssortments = data.shape[0]
counter = -1
prodid = 0
alldata = []
prodIDs = {}
for row in data:
    counter += 1
    # Για όλες τις γραμμές των δεδομένων (assortments)
    if counter <= howManyAssortments:
        # Ξεκινώντας από την 1η στήλη, το πρόγραμμα διαβάζει ανά 4 στήλες
        # ώστε να απομονώσει την τετράδα των δεδομένων κάθε κωδικού
        for i in range(0, 4 * ProductsPerAssortment, 4):  # 4 * 7
            # Βοηθητική λίστα
            helperrow = []
            helperrow.append(counter)
            # Η λίστα sliced περιέχει την τετράδα δεδομένων για τον
            # εκάσττοε κωδικό προϊόντος που επεξεργάζεται το πρόγραμμα.
            sliced = row[i:i + 4]
            # Η πρώτη θέση της τετράδας αντιστοιχεί στο Product ID
            current_prod_id = sliced[0]
            # Δημιουργείται λεξικό με τα Product IDs
            prodIDs[current_prod_id] = 1
            # Η 4η θέση περιέχει τη συνεισφορά του κωδικού στα έσοδα
            current_contr_per = sliced[3]
            # Απομονώνει κάθε τετράδα σε μία ανεξάρτητη γραμμή, ώστε να μπορεί
            # να δουλέψει το μοντέλο.
            for item in sliced:
                helperrow.append(item)
            # Συνολικά έσοδα της τρέχουσας γραμμής
            current_revenue = row[4 * ProductsPerAssortment]
            # Στην τελευταία στήλη εισάγεται το συνολικό έσοδο του καλαθιού
            helperrow.append(current_revenue)
            # Δημιουργείται μία ακόμη στήλη όπου υπολογίζεται το έσοδο από
            # τον κωδικό προϊόντος της τρέχουσας γραμμής
            helperrow.append(current_contr_per / 100 * current_revenue)
            # Η γραμμή προστίθεται σε ένα dataframe το οποίο ενσωματώνει τις επεξεργασμένες
            # γραμμές τόλων των καλαθιών του αρχείου
            alldata.append(helperrow)

# Υπολογισμός του συνολικού πλήθους προϊόντων. Αυτό γίνεται
# μετρώντας το πλήθος των κλειδιών του λεξικού, στο οποίο εισάγουμε τους μοναδικούς
# κωδικούς των προϊόντων που επεξεργαστήκαμε
TotalProducts = len(prodIDs.keys())

# Δημιουργούμε ένα βοηθητικό CSV αρχείο στο οποίο καταχωρούμε το επεξεργασμένο dataframe
# με τα δεδομένα κάθε προϊόντος σε μία γραμμή.
fields = ['AssortmentID', 'ID', 'Cust_Perc', 'Excl_perc', 'Contrib_perc', 'Revenue', 'Prod_revenue']
with open(datafolder + 'assortments.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(alldata)

# Διαβάζουμε το αρχείο για να δημιουργηθεί ένα Pandas DataFrame
# το οποίο έχει τις εξής στήλες:
# 1. Assortment ID του καλαθιού. Στο αρχείο υπάρχουν τόσες γραμμές με το ίδιο Assortment ID
#    όσα και τα προϊόντα που πουλήθηκαν στο καλάθι.
# 2. ID: ο κωδικός του πρϊόντος
# 3. Cust_Perc:
# 4. Excl_perc: ποσοστό πελατών που αγόρασαν αποκλειστικά το προϊόν
# 5. Contrib_perc: Ποσοστό συμμετοχής του προϊόντος στα έσοδα
# 6. Revenue: τα συνολικά έσοδα του καλαθιού με το συγκεκριμένο Assortment ID
# 7. Prod_revenue: τα έσοδα που αντιστοιχούν στο προϊόν της γραμμής.
data = pd.read_csv(datafolder + 'assortments.csv', delimiter=',')
data = data.astype({"ID": int})
print(data.head(10))
# Ομαδοποίηση με την εντολή groupby των εγγραφών του dataframe ανά Assortment ID kai ID (προϊόντος)
# Η ομαδοποίηση εκτελεί άθροιση στη στήλη Prod_revenue
baskets = data.groupby(['AssortmentID', 'ID'])['Prod_revenue',].sum().unstack().reset_index().fillna(0).set_index(
    'AssortmentID')
print(baskets.head(10))
# Δημιουργούμε ένα βοηθητικό CSV αρχείο, ώστε να μπορέσουμε στη συνέχεια
# να εξαιρέσουμε ορισμένες ακρότατες τιμές.
baskets.to_csv(datafolder + 'OUT.CSV')
# Αγνοούμε την 1η και 3η γραμμή του αρχείου, γιατί δεν περιέχουν χρήσιμα δεδομένα
# Μόνο ετικέτες τις οποίες και αγνοούμε
baskets = pd.read_csv(datafolder + 'OUT.CSV', skiprows=[0, 2])
print(baskets.head(10))
# Υπολογίζουμε τη συσχέτιση των στηλών του dataframe που δημιουργήθηκε.
# Παρατηρούμε έτσι ποιο προϊόν πωλείται μαζί με κάποιο άλλο και πόσο
# ισχυρή είναι αυτή η συσχέτιση, χρησιμοποιώντας χρωματικό κώδικά.
corr = baskets.corr()
# Δημιουργία ενός heatmap, όπου απεικονίζεται η παραπάνω συσχέτιση
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=0.8, square=True)
plt.show()

# Δημιουργείται ένα διάγραμμα γραμμής, όπου απεικονίζεται
# η κατανομή των εσόδων ανά προϊόν στα καλάθια που επεξεργαζόμαστε
plt.figure(figsize=(8, 8))
plt.title('Prod_revenue Distribution Plot')
sns.distplot(data['Prod_revenue'])
plt.show()

# Δημιουργείται ένα ιστόγραμμα, όπου απεικονίζεται
# η κατανομή των εσόδων ανά προϊόν στα καλάθια που επεξεργαζόμαστε
data.hist('Prod_revenue')
plt.show()

# Έλεγχος για επιτρεπτές τιμές κωδικών προϊόντων.
# Αποδεκτές τιμές είναι: > 0 και < TotalProducts
productNo = -1
while (productNo < 0) or (productNo > TotalProducts - 1):
    productNo = int(input('Select a Product code from (0...' + str(TotalProducts - 1) + '): '))

# Αφαιρούμε τη στήλη του προϊόντος για το οποίο αναζητούμε την πρόβλεψη
X = baskets.drop(baskets.columns[productNo], axis=1)
# Δημιουργούμε dataframe με μία στήλη (με τίτλο το κωδικό του προϊόντος που επεξεργαζόμαστε)
# και το σύνολο των εσόδων αυτού του προϊόντος.
y = baskets[baskets.columns[productNo]]
# train test spit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)

# Υπολογισμός του μέσου όρου των πραγματικών πωλήσεων του προϊόντος.
# Χρησιμοποιείται στο τελευταίο ερώτημα, όπου συγκρίνουμε μέσους όρους
# πραγματικών εσόδων και των προβλέψεων που παρήγαγε το μοντέλο.
averageRevenue = np.mean(y_train)

# Δημιουργία τριών λεξικών. Σε καθένα από αυτά αποθηκεύονται:
# Τα λεξικά έχουν σαν κλειδί το λεκτικό του μοντέλου που εκτελείται κάθε φορά
# Σε κάθε ένα λεξικό αποθηκεύουμε:
# r2_results: Το αποτέλεσμα r2 που προκύπτει για το μοντέλο
r2_results = {}
# mape_results: Το αποτέλεσμα MEAN AVERAGE PERCENTAGE ERROR για το αντίστοιχο μοντέλο
mape_results = {}
# predictions_results: Αποθηκεύει τα αποτελέσματα της πρόβλεψης
predictions_results = {}

# Ερώτημα α1
# Ξεκινά η εκτέλεση των μοντέλων
# 1o ΜΟΝΤΕΛΟ
# GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=100)

# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'n_estimators': [100, 500, 1000],
              'learning_rate': [0.2, 0.15, 0.1],
              'max_depth': [2, 3, 4, 6],
              'min_samples_leaf': [1, 3, 5]}

grid = GridSearchCV(gbr, param_grid, refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)

# Best parameter after hyper parameter tuning
print('=========================== GradientBoostingRegressor ================================')
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['GradientBoostingRegressor'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['GradientBoostingRegressor'] = mape
predictions_results['GradientBoostingRegressor'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")
'''
# 2o ΜΟΝΤΕΛΟ
# RandomForestRegressor
rfr = RandomForestRegressor()

# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'n_estimators': [100, 500, 1000],
              'criterion': ['mse', 'mae'],
              'max_depth': [2, 3, 4, 6],
              'min_samples_leaf': [1, 3, 5]}

grid = GridSearchCV(rfr, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('============================= RandomForestRegressor ===============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['RandomForestRegressor'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['RandomForestRegressor'] = mape
predictions_results['RandomForestRegressor'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")
'''
# 3o ΜΟΝΤΕΛΟ
# LinearRegression
lr = LinearRegression()

# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'fit_intercept': [True],
              'normalize': [False],
              'copy_X': [True],
              'positive': [False]}

grid = GridSearchCV(lr, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('============================ LinearRegression ============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['LinearRegression'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['LinearRegression'] = mape
predictions_results['LinearRegression'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")

# 4o ΜΟΝΤΕΛΟ
# Lasso
lasso = Lasso()

# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'alpha': [0.05, 0.1, 0.2, 0.5, 1.0],
              'max_iter': [100000],
              'random_state': [1],
              'positive': [True, False]}

grid = GridSearchCV(lasso, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('=========================== Lasso ============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['Lasso'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['Lasso'] = mape
predictions_results['Lasso'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")

# 5o ΜΟΝΤΕΛΟ
# Elastic Net
ENet = ElasticNet()
# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'alpha': [0.05, 0.1, 0.2, 0.5, 1.0],
              'l1_ratio': [0.2, 0.5, 0.8],
              'random_state': [1, 3]}

grid = GridSearchCV(ENet, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('============================ ElasticNet ===============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['ElasticNet'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['ElasticNet'] = mape
predictions_results['ElasticNet'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")

# 6o ΜΟΝΤΕΛΟ
# Ridge
ridge = Ridge()
# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'alpha': [0.0005, 0.5, 1.0]}

grid = GridSearchCV(ridge, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('============================= Ridge ===============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['Ridge'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['Ridge'] = mape
predictions_results['Ridge'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")

# 7o ΜΟΝΤΕΛΟ
# AdaBoostRegressor
adaboost = AdaBoostRegressor()
# Καθορίζουμε παραμέτρους του μοντέλου για να τρέχουν οι συνδυασμοί
param_grid = {'n_estimators': [25, 50]}

grid = GridSearchCV(adaboost, param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(X_train, y_train)
print('========================== AdaBoostRegressor ===============================')
# Best parameter after hyper parameter tuning
print("grid.best_params: ", grid.best_params_)
# Model Parameters
print("grid.best_estimator: ", grid.best_estimator_)
# Prediction using best parameters
grid_predictions = grid.predict(X_test)
print("PREDICTIONS:")
print(grid_predictions)
print("prediction using best params")
r2 = r2_score(y_test, grid_predictions)
r2_results['AdaBoostRegressor'] = r2
mape = mean_absolute_percentage_error(y_test, grid_predictions)
mape_results['AdaBoostRegressor'] = mape
predictions_results['AdaBoostRegressor'] = grid_predictions
print("r2 score : ", r2)
print("MAPE     : ", mape," %")

# Εύρεση του μοντέλου με το μεγαλύτερο r2
# και μικρότερο MAPE
max_r2 = -1
for key in r2_results.keys():
    if r2_results[key] > max_r2:
        max_r2 = r2_results[key]
        max_r2_label = key
print('Model with best r2:     ', max_r2_label)
print('Max r2 value:           ', max_r2)

min_mape = mape_results['Ridge']
min_mape_label = 'Ridge'
for key in mape_results.keys():
    if mape_results[key] < min_mape:
        min_mape = mape_results[key]
        min_mape_label = key
print('Model with lowest MAPE: ', min_mape_label)
print('Min MAPE value:         ', min_mape," %")

print('REVENUE FORECAST FOR PRODUCT: ' + str(productNo))
print(predictions_results[max_r2_label])

labels = []
predictions = predictions_results[max_r2_label].tolist()
actuals = y_train.tolist()
for item in y_train:
    labels.append('Actual')
revenue_list = actuals + predictions
for item in predictions:
    labels.append('Prediction')
data = pd.DataFrame({'Revenue': revenue_list, 'Prediction': labels})
data.insert(0, 'ID', data.index + 1)
sns.lineplot(x='ID', y='Revenue', data=data, hue="Prediction")
plt.show()

# Ερώτημα α2
# Υπολογισμός μέσου όρου των προβλέψεων
PredictedAverage = np.mean(predictions_results[max_r2_label])
print('Average vs Predicted average: ' + str(averageRevenue) + ' vs ' + str(PredictedAverage))