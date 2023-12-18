# import pandas as pd
# data = {
#     'Name': ['Antony', 'Joshy', 'Leo', 'Harold', 'Parthi'],
#     'Age': [55, 40, 35, 45, 35],
#     'Salary': [5000000, 70000, 90000, 120000, 150000]
# }
# df = pd.DataFrame(data)
# df.set_index('Name', inplace=True)
# df_sorted = df.sort_values('Age')
# print(df_sorted)

# import pandas as pd
# df = pd.read_csv('D:/CSE/Academic/Sem VII/DS/lab/archive/a.csv')
# miss=df.isnull().sum()
# print("Count: ")
# print(miss)
#
# print(df.head())
# ind_df= df.set_index(['ID','State','Variable','% Change from 2012'])
# print(ind_df.head())

# import pandas as pd
# import numpy as py
#
# def create_pivot_table(df, index, columns, values, aggfunc):
#     pivot_table = df.pivot_table(
#         index=index,
#         columns=columns,
#         values=values,
#         aggfunc=aggfunc,
#     )
#
#     return pivot_table
# df = pd.DataFrame({
#     "product": ["apple", "banana", "orange", "apple", "banana", "orange"],
#     "region": ["USA", "USA", "USA", "EU", "EU", "EU"],
#     "sales": [100, 200, 300, 400, 500, 600],
# })
# pivot_table = create_pivot_table(df, index=["region"], columns=["product"], values=["sales"], aggfunc="sum")
# print(pivot_table)
#
# def perform_mathematical_operation(expression):
#    return eval(expression)
# expression = input("Enter a mathematical expression: ")
# result = perform_mathematical_operation(expression)
# print(f"The result of the expression is {result}")

# import pandas as pd
#
# def query_dataframe(df, conditions):
#         filtered_df = df.query(conditions)
#
#         print(filtered_df)
#
# df = pd.DataFrame({
#     "name": ["Antony", "Leo", "Harold", "Dutt", "Elisa"],
#     "age": [25, 30, 35, 40, 45],
#     "occupation": ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Sales Manager"],
# })
# conditions = input("Enter the conditions for the query: ")
# query_dataframe(df, conditions)

# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('D:/CSE/Academic/Sem VII/DS/lab/archive/a.csv')
# x_column = df['2013']
# y_column = df['2012']
# plt.scatter(x_column, y_column)
# plt.title('Scatter Plot')
# plt.xlabel('X-axis')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(123)
data = pd.DataFrame(np.random.rand(50, 3), columns=['X', 'Y', 'Z'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['X'], data['Y'], data['Z'], c='r', marker='o')
Setting labels for each axis
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

Adding a title
ax.set_title('3D Scatter Plot')

Displaying the 3D plot
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('D:/CSE/Academic/Sem VII/DS/lab/likes/Instagram_Likes1.csv')
# num_likes = data['Likes']
# emotions = data['Emotions']
# average_likes = num_likes.mean()
# emotion_counts = emotions.value_counts()
# print("Average number of likes per post:", average_likes)
# print("Emotion counts:")
# print(emotion_counts)
# plt.figure(figsize=(10, 6))
# plt.plot(num_likes)
# plt.xlabel("Post number")
# plt.ylabel("Number of likes")
# plt.title("Number of Likes per Post")
# plt.grid(True)
# plt.show()
#
# # Plot the emotion counts
# plt.figure(figsize=(8, 6))
# plt.bar(emotion_counts.index, emotion_counts.values)
# plt.xlabel("Emotion")
# plt.ylabel("Count")
# plt.title("Emotion Counts")
# plt.grid(True)
# plt.show()

import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk import ne_chunk, pos_tag
from nltk.data import load

# Load the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the named entity recognizer
ner_tagger = load('nltk:english_ner_ontonotes5')

# Read the CSV file into a DataFrame
data = pd.read_csv('D:/CSE/Academic/Sem VII/DS/lab/twitter/twitter_training.csv')

# Perform sentiment analysis on each tweet
for tweet in data['tweet']:
    # Tokenize the tweet
    tokens = word_tokenize(tweet)

      Tag the tokens with their POS tags
    pos_tags = pos_tag(tokens)

    # Chunk the POS-tagged tokens into named entities
    chunked_tokens = ne_chunk(pos_tags)

    # Convert the chunked tokens to a tree
    tree = conlltags2tree(chunked_tokens)

    # Extract named entities from the tree
    named_entities = []
    for subtree in tree:
        if subtree.label() == 'NE':
            named_entities.append(subtree[0])

    # Perform sentiment analysis on the tweet
    sentiment = analyzer.polarity_scores(tweet)
    compound = sentiment['compound']

    # Display the sentiment score and named entities
    print("Tweet:", tweet)
    print("Sentiment score:", compound)
    print("Named entities:", named_entities)
    print("---------------------------------")
