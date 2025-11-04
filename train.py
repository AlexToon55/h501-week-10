import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor


# Exercise 1 

# save model as 'model_1.pickle'
def save_model(model, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)

# load coffee data from URL
def load_data():
    coffeeURL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(coffeeURL)
    return df


# train and save model
df = load_data()
df = df.dropna(subset=['100g_USD', 'rating'])
x = df[['100g_USD']]
y = df['rating']
print(x.shape, y.shape)

model1 = LinearRegression()
model1.fit(x, y)
save_model(model1, 'model_1.pickle')

print(sorted(df['roast'].dropna().unique().tolist()))




# Exercise 2

roast_map = {
    'Light': 0,
    'Medium-Light': 1,
    'Medium': 2,
    'Medium-Dark': 3,
    'Dark': 4
}
df['roast_num'] = df['roast'].map(roast_map)

df2 = df.dropna(subset=['100g_USD', 'rating', 'roast_num'])

X2 = df2[['100g_USD', 'roast_num']]
y2 = df2['rating']

print(X2.shape, y2.shape)

model2 = 
print(df[['roast', 'roast_num']].head())
print("n_missing:", df['roast_num'].isna().sum())