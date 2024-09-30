import streamlit as st
import pandas as pd
import pickle

# open csv
df = pd.read_csv(r'C:\Users\Edwin\Python\bootcamp\Projects\SVM\csv\Student_Marks.csv')

st.title('Support Vector Machine (SVM) Regression with Student Marks')
st.write("This project focuses on conducting exploratory analysis to understand how specific variables impact student performance using Support Vector Machines (SVM). Additionally, it serves as an application for predicting student grades based on the number of subjects selected and the hours dedicated to studying.")

st.subheader('a) Data Exploration')
quantile_code = '''
import pandas as pd
df['Performance'] = pd.qcut(df['Marks'], q=4, labels= ['Poor', 'Mid', 'Satisfactory', 'Good'])
'''
st.code(quantile_code, language="python")

st.image(r'C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\mark.png')

hist_code = '''
def marks():
    sns.set_style('whitegrid')
    sns.displot(df['Marks'], kde = True, color ='red', bins = 30)
    plt.title('Student Marks Distribution')
    plt.savefig('./images/student.png')

marks()
'''
st.code(hist_code, language='python')
st.write("The chart above is a histogram that illustrates the distribution of students' marks, accompanied by a line chart for additional insights. The graph is left-skewed, indicating that, overall, students tend to perform poorly with lower scores.")

st.write("To examine the impact of various factors on students' marks, here is a pairplot that illustrates the correlations between different variables and the outcomes.")
st.image(r'C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\pairplot.png')
pair_code = '''
def pairplot():
    sns.pairplot(df, hue="Performance", hue_order = ['Poor', 'Mid', 'Satisfactory', 'Good'], height=2.5)
    plt.savefig('./images/pairplot.png')

pairplot()
'''
st.code(pair_code, language= 'python')
st.write("The dataset includes just two variables: the number of courses and study time. The plot demonstrates how these factors contribute to student marks. Notably, there is a strong positive and linear correlation between study time and student marks. Additionally, the data points are grouped according to student performance categories, as previously mentioned.")

st.write("The plot below is a correlation matrix that illustrates the relationship between each variable and the output. It reveals a positive correlation of 0.42 between the number of subjects and student marks. More significantly, there is a very strong correlation of 0.94 between study hours and student marks, indicating that the more hours a student studies, the higher their marks tend to be.")
st.image(r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\corr_matrix.png")
corr_code = """
def corr_matrix():
    corr_df = df[['number_courses', 'time_study', 'Marks']]
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=.8, cmap='coolwarm')
    plt.title('Correlation Matrix of Student Marks')
    plt.savefig('./images/corr_matrix.png')

corr_matrix()
"""
st.code(corr_code, language= 'python')

st.write("Focusing on the correlation between study hours and student marks, a scatter plot has been created to illustrate this relationship. In the plot, the data points are categorized based on student performance levels, providing a clearer visualization of how different levels of study hours influence marks.")
st.image(r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\jointplot.png")

join_code = """
def jointplot():
    sns.jointplot(df, x="time_study", y="Marks", hue="Performance", hue_order = ['Poor', 'Mid', 'Satisfactory', 'Good'])
    plt.savefig('./images/jointplot.png')

jointplot()
"""
st.code(join_code, language= 'python')

st.write("Finally, a regression analysis was conducted to create a trend line that represents the relationship between study hours and student marks. The plot displays both the slope and the R-squared value, indicating a strong positive correlation between these variables.")
st.image(r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\regression.png")
reg_code = """
def regression():
    p = sns.regplot(data=df, x="time_study", y="Marks")
    slope, intercept, r, p, sterr = scipy.stats.linregress(df['time_study'], df['Marks'])
    r_squared = r ** 2
    plt.text(2, 45, 'y = ' + str(round(slope,3)) + 'x + '+ str(round(intercept,3)))
    plt.text(2, 40, 'R^2 =' + str(round(r_squared,3)))
    plt.title('The regression analysis for student marks')
    plt.savefig('./images/regression.png')

regression()
"""

st.code(reg_code, language='python')

st.subheader('b) Model Selection & Data Preprocessing')
st.write("Since the data consists of continuous values, it is advisable to use Support Vector Regression (SVR) for modeling. SVR effectively captures the relationship between the input features and continuous outcomes, accommodating non-linear trends through the use of kernel functions. In this step, the data is divided into X_train and y_train to facilitate model training.")

select_code ="""
# Extract X & y variables 
X = df.drop(['Marks', 'Performance'], axis= 1)
y = df['Marks']

# Import train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
st.code(select_code, language='python')

st.write("Next, feature scaling is necessary, and this step demonstrates how to normalize the data before fitting it into the model.")
feat_code = '''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
st.code(feat_code, language='python')

st.subheader('c) Model Training & Evaluation')
st.write("The model is currently trained using the SVM algorithm, incorporating variables to predict responses for both the training and test datasets.")
build_code = '''
# Import svm model
from sklearn import svm
reg = svm.SVR(kernel='linear') # Linear Kernel
reg.fit(X_train, y_train)

# Predict the response for both train and test dataset
org_ytrain_pred = reg.predict(X_train)
org_ytest_pred = reg.predict(X_test)
'''
st.code(build_code,language='python')

st.write("The function below illustrates how the model is evaluated using performance metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Ultimately, the results will be saved as a DataFrame and exported to CSV format for data visualization purposes.")

acc_code = '''
# Evaluating the model (MAE, MSE, RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

def score(train_pred,test_pred,  is_optimized=False):
    train_dict = {
        'metrics': ['MAE', 'MSE', 'RMSE'],
        'Training': [
            np.round(mean_absolute_error(y_train, train_pred), 2),
            np.round(mean_squared_error(y_train, train_pred), 2),
            np.round(root_mean_squared_error(y_train, train_pred), 2)
        ],
        'Testing': [
            np.round(mean_absolute_error(y_test, test_pred), 2),
            np.round(mean_squared_error(y_test, test_pred), 2),
            np.round(root_mean_squared_error(y_test, test_pred), 2),
        ]
    }
'''
st.code(acc_code, language='python')

st.image(r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\org_metrics.png")
metrics_code = '''
def model_performance(df, is_optimised = False):
    # Set the bar width and positions
    bar_width = 0.35
    index = np.arange(len(df['metrics']))
    
    # Create bars for Training and Testing
    plt.bar(index, df['Training'], bar_width, label='Training', alpha=0.7, edgecolor = 'black')
    plt.bar(index + bar_width, df['Testing'], bar_width, label='Testing', alpha=0.7, edgecolor = 'black')

    # Adding labels 
    plt.xlabel('Metrics')
    plt.ylabel('Error')
    plt.xticks(index + bar_width / 2, df['metrics'])  # Center the tick labels
    plt.legend()  # Show the legend
    plt.tight_layout()
'''
st.code(metrics_code, language='python')
st.write("The bar plot above displays the error rates for the specified metrics, as outlined in the previous code. Hyperparameter tuning will be conducted to minimize these errors. The code used to assess the model's performance in the original setup remains unchanged after hyperparameter tuning, which will be demonstrated next.")

st.subheader('d) Hyper-Parameter Tuning')
st.write('To minimize the errors effectively, GridSearchCV is utilized to identify the optimal parameters for error reduction.')
hp_code = '''
# Try some hyperparameter tuning
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear', 'poly', 'sigmoid']}

grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
grid.best_params_ # {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
  
# print how our model looks after hyper-parameter tuning 
grid.best_estimator_ # SVR(C=1000, gamma=0.01)
'''
st.code(hp_code, language='python')
st.write("The GridSearchCV results indicate that the best parameters for the SVM model are {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}. Following a similar approach to the original model, a tuned version is implemented to further reduce the error.")

tuned_code = '''
# Create a tuned model for predictions
tuned_reg = svm.SVR(C = 1000, gamma = 0.01, kernel='rbf')
tuned_reg.fit(X_train, y_train)

# Predict the response for both train and test dataset after tunning
new_ytrain_pred = tuned_reg.predict(X_train)
new_ytest_pred = tuned_reg.predict(X_test)
'''
st.code(tuned_code, language= 'python')

st.image(r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\images\tuned_metrics.png")
st.write("Here is the model's performance following hyperparameter tuning. As illustrated in the figure, the error decreases significantly. The optimized model will be saved for future predictions of student performance.")

st.subheader('e) Prediction App')
st.write("Introducing a student mark prediction app that takes into account the number of subjects and study hours. You can adjust the sliders to estimate the predicted student mark.")

# Save the input features
def input_features():
    courses = st.slider('Number of Courses', 3, 8)
    time = st.slider('Hours of Studying', 0, 8)

    data = {
        'number_courses': courses,
        'time_study': time,
    }
    features = pd.DataFrame(data, index=[0])
    return features

def performance(mark):
    if mark >= df['Marks'].quantile(q=0.75):
        comment = 'Good'
    elif df['Marks'].quantile(q=0.5) <= mark < df['Marks'].quantile(q=0.75):
        comment = 'Satisfactory'
    elif df['Marks'].quantile(q=0.25) <= mark < df['Marks'].quantile(q=0.5):
        comment = 'Mid'
    else:
        comment = 'Poor'
    return comment

# Open the saved models
pickle_path = r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\model\tuned_svm_model.pkl"
pickle_scaler_path = r"C:\Users\Edwin\Python\bootcamp\Projects\SVM\model\scaler.pkl"

with open(pickle_path, 'rb') as file:
    saved_model = pickle.load(file)

with open(pickle_scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

user_df = input_features()

if st.button('Predict'):
    # Scale the input features  
    user_df_scaled = scaler.transform(user_df)

    prediction = saved_model.predict(user_df_scaled)
    st.write(f'Predicted Output (Student Score): {prediction[0]:,.1f}')  
    st.write(f'Student Performance: {performance(prediction)}')