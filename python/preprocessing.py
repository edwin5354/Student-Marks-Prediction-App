import pandas as pd

df = pd.read_csv('./csv/Student_Marks.csv')

# Shape of the data
df.shape # (100, 3)

# print data types
df.info() # num_courses: int64 others float64

# see descriptive statistics that summarize the central tendency
df.describe()

# Check null values
df.isnull().any() # There is not any null value

# Create a new group as top/ middle and bottom students --> Bin by Quantile 
df['Performance'] = pd.qcut(df['Marks'], q=4, labels= ['Poor', 'Mid', 'Satisfactory', 'Good'])

df.to_csv('./csv/processed.csv', index=False)