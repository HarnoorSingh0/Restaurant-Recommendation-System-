import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import warnings
from flask_ngrok import run_with_ngrok
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import linear_kernel
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import r2_score
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords


"""
Preprocessing the data

"""

df=pd.read_csv('zomato.csv',  encoding = 'ISO-8859-1',  low_memory = False, nrows = 10000)

#Removing url and phone column and also the column of dish_liked as we can't afford these many values gone
df.drop(columns=['url','phone','dish_liked'],axis=1,inplace=True)
df.dropna(axis=0,inplace=True)
df.drop_duplicates(inplace=True)
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'],errors = 'coerce')
#Changing the names to ease
df=df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})
#Removing 5 from rate
df['rate']=df['rate'].apply(lambda x:x.replace('/5',''))
df['rate'] = pd.to_numeric(df['rate'],errors = 'coerce')

restaurants = list(df['name'].unique())
df['Mean Rating'] = 0

for i in range(len(restaurants)):
    df['Mean Rating'][df['name'] == restaurants[i]] = df['rate'][df['name'] == restaurants[i]].mean()
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
df[['Mean Rating']] = scaler.fit_transform(df[['Mean Rating']]).round(2)
df['online_order']=df['online_order'].replace(('Yes','No'),(1,0))
df['book_table']=df['book_table'].replace(('Yes','No'),(1,0))
df.drop('address',axis=1,inplace=True)
df.dropna(axis=0,inplace=True)
df['reviews_list']=df['reviews_list'].apply(lambda x:x.replace('\\n',''))
df['menu_item']=df['menu_item'].apply(lambda x:x.replace('\\n',''))
df['reviews_list']=df['reviews_list'].str.lower().str.replace('[^\w\s]','')
df['cuisines']=df['cuisines'].str.lower().str.replace('[^\w\s]','')
df['menu_item']=df['menu_item'].str.lower().str.replace('[^\w\s]','')

####################################


stop=set(stopwords.words('english'))
def remove_stop(text):
    return " ".join([i for i in str(text).split() if i not in stop])


df['reviews_list']=df['reviews_list'].apply(lambda text:remove_stop(text))
df['cuisines']=df['cuisines'].apply(lambda text:remove_stop(text))
df['menu_item']=df['menu_item'].apply(lambda text:remove_stop(text))

def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]



df=df.drop(['menu_item', 'votes'],axis=1)

# Randomly sample 60% of your dataframe
df_percent = df.sample(frac=0.5)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new


"""
Creating the webapp
"""
app= Flask(__name__, template_folder='templates')
run_with_ngrok(app)
@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict', methods=['GET', 'POST'])
def main():

    if request.method == 'POST':
        m_name = request.form['name']
        m_name = m_name.title()


        result_final = recommend(m_name)
        names = []
        cuisines = []
        rating = []
        price = []
        for i in range(len(result_final)):
            names.append(result_final.index[i])
            cuisines.append(result_final.iloc[i][0])
            rating.append(result_final.iloc[i][1])
            price.append(result_final.iloc[i][2])


            return render_template('index.html', names="You may like {}".format(names),cuisines="It's cuisine : {}".format(cuisines),rating = "It's rating : {}".format(rating),price= "price : {}".format(price), search_name=m_name)

    return render_template("index.html")


if __name__ == "__main__":
    
    app.run()