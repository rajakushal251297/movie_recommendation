import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# credits=pd.read_csv("credits.csv")
# movies=pd.read_csv("movies.csv")
# movies=movies.merge(credits, on="title")
# movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]
# movies.dropna(inplace=True)

# def convert(obj):
#     L=[]
#     for i in ast.literal_eval(obj):
#         L.append(i["name"])
#     return L
# movies["genres"]=movies["genres"].apply(convert)
# movies["keywords"]=movies["keywords"].apply(convert)

# def get_actor(obj):
#     L=[]
#     count=0
#     for i in ast.literal_eval(obj):
#         if(count<3):
#             L.append(i["name"])
#             count=count+1
#         else :
#             break
#     return L

# movies["cast"]=movies["cast"].apply(get_actor)

# def get_director(obj):
#     L=[]
#     for i in ast.literal_eval(obj):
#         if i["job"]=="Director":
#             L.append(i["name"])
#             break
#     return L


# movies["crew"]=movies["crew"].apply(get_director)
# movies["overview"]=movies["overview"].apply(lambda x: x.split())

# movies["keywords"]=movies["keywords"].apply(lambda x: [i.replace(" ","") for i in x])
# movies["genres"]=movies["genres"].apply(lambda x: [i.replace(" ","") for i in x])
# movies["cast"]=movies["cast"].apply(lambda x: [i.replace(" ","") for i in x])
# movies["crew"]=movies["crew"].apply(lambda x: [i.replace(" ","") for i in x])

# movies["tags"]=movies["overview"]+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]

# movies=movies[["movie_id","title","tags"]]



# movies["tags"]=movies["tags"].apply(lambda x: " ".join(x))
# movies["tags"]=movies["tags"].apply(lambda x:x.lower())
# df=pd.DataFrame(movies)
# df.to_csv("movies_one.csv")

movies=pd.read_csv("movies_one.csv")
cv =CountVectorizer(max_features=5000,stop_words="english")
vectors=cv.fit_transform(movies["tags"]).toarray()
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies["tags"].apply(stem)
similarity=cosine_similarity(vectors)
r=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
lst=[]
def recommend(movie):
    movie_index=movies[movies["title"]==movie].index[0]
    distances =similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        # print(movies.iloc[i[0]].title)
        lst.append(movies.iloc[i[0]].title)
    return lst
        
movie_name=movies["title"]
list_of_movie_name=list(movie_name)
    
    
# for i in recommend("Avatar"):
#         print(i)
# recommend("The Dark Knight Rises")




st.title("Movie Recommender System")
option =st.selectbox("Search",list_of_movie_name)

if st.button("Search"):
    for i in recommend(option):
        st.write(i)