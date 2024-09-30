import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

df = pd.read_csv('./csv/processed.csv')
org_metrics = pd.read_csv('./csv/org_svm_metrics.csv')
tuned_metrics = pd.read_csv('./csv/tuned_svm_metrics.csv')

def marks():
    sns.set_style('whitegrid')
    sns.displot(df['Marks'], kde = True, color ='red', bins = 30)
    plt.title('Student Marks Distribution')
    plt.savefig('./images/student.png')

def pairplot():
    sns.pairplot(df, hue="Performance", hue_order = ['Poor', 'Mid', 'Satisfactory', 'Good'], height=2.5)
    plt.savefig('./images/pairplot.png')

def corr_matrix():
    corr_df = df[['number_courses', 'time_study', 'Marks']]
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=.8, cmap='coolwarm')
    plt.title('Correlation Matrix of Student Marks')
    plt.savefig('./images/corr_matrix.png')

def jointplot():
    sns.jointplot(df, x="time_study", y="Marks", hue="Performance", hue_order = ['Poor', 'Mid', 'Satisfactory', 'Good'])
    plt.savefig('./images/jointplot.png')

#jointplot()

def regression():
    p = sns.regplot(data=df, x="time_study", y="Marks")
    slope, intercept, r, p, sterr = scipy.stats.linregress(df['time_study'], df['Marks'])
    r_squared = r ** 2
    plt.text(2, 45, 'y = ' + str(round(slope,3)) + 'x + '+ str(round(intercept,3)))
    plt.text(2, 40, 'R^2 =' + str(round(r_squared,3)))
    plt.title('The regression analysis for student marks')
    plt.savefig('./images/regression.png')

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

    if not is_optimised:
        plt.title('Original SVM Model Regression Training vs Testing Metrics')
        plt.ylim([0, 15])
        plt.savefig('images/org_metrics.png')
    else:
        plt.title('SVM Model Regression Training vs Testing Metrics After Tuning')
        plt.ylim([0, 0.5])
        plt.savefig('images/tuned_metrics.png')

#print(df['Marks'].quantile(q=0.75))