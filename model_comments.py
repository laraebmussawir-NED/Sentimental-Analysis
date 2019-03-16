import pandas as pd
import re

df1 = pd.read_csv('kinza.csv')
df2 = pd.read_csv('laraeb.csv')
df1.head()
df1[df1['Label'] == 1]
df1[df1['Label'] == 0]
df1[df1['Label'] == -1]

df2[df2['Label'] == 1]
df2[df2['Label'] == 0]
df2[df2['Label'] == -1]

positive_comments = []
negative_comments = []
neutral_comments = []

positive_comments.extend(df1[df1['Label'] == 1].comments)
positive_comments.extend(df2[df2['Label'] == 1].comments)

negative_comments.extend(df2[df2['Label'] == -1].comments)
negative_comments.extend(df1[df1['Label'] == -1].comments)

neutral_comments.extend(df2[df2['Label'] == 0].comments)
neutral_comments.extend(df1[df1['Label'] == 0].comments)

def clean(text):
    try:
        text = re.split("""b['"]""", text)[1]
        text = re.sub(r'\\x[a-f0-9A-F][a-f0-9A-F]', '', text)
        text = re.sub(r'[^a-zA-Z ]', "", text)
        text = re.sub(r'\\n', ' ', text)
        text = re.sub(r'\\r', ' ', text)
        return text.lower()
    except:
        print(text)
        pass

cleaned_positive_comments = [clean(comment) for comment in positive_comments]
cleaned_negative_comments = [clean(comment) for comment in negative_comments]
cleaned_neutral_comments = [clean(comment) for comment in neutral_comments]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))


def reduce_length(text):
    pattern = re.compile(r'(.)\1{2,}')
    return pattern.sub(r'\1\1', text)


for x in range(0, len(cleaned_positive_comments)):
    word_token = word_tokenize(cleaned_positive_comments[x])
    words = [w for w in word_token if w not in stop_words]
    cleaned_positive_comments[x] = ' '.join(words)

for x in range(0, len(cleaned_negative_comments)):
    word_token = word_tokenize(cleaned_negative_comments[x])
    words = [w for w in word_token if w not in stop_words]
    cleaned_negative_comments[x] = ' '.join(words)

for x in range(0, len(cleaned_neutral_comments)):
    word_token = word_tokenize(cleaned_neutral_comments[x])
    words = [w for w in word_token if w not in stop_words]
    cleaned_neutral_comments[x] = ' '.join(words)

positive_df = pd.DataFrame(columns=['text'], data=cleaned_positive_comments)
positive_df['sentiment'] = 1

negative_df = pd.DataFrame(columns=['text'], data=cleaned_negative_comments)
negative_df['sentiment'] = -1

neutral_df = pd.DataFrame(columns=['text'], data=cleaned_neutral_comments)
neutral_df['sentiment'] = 0

data = pd.DataFrame()
data = data.append(positive_df)
data = data.append(negative_df)
data = data.append(neutral_df)

names = pd.read_csv('BankAlfalahOfficial_Comments1.csv')
names = names['User_Name']
names_list = []
for name in names:
    names_list.extend(name.split(' '))


words = []
for comment in data.text:
    word = comment.split(' ')
    for w in word:
        if w not in words:
            words.append(w)
import numpy as np
words_df = pd.DataFrame(columns=['count'], index=words, data=np.zeros(len(words)))


for comment in data.text:
    com_words = comment.split(' ')
    for w in com_words:
        if w in words:
            words_df.loc[w] += 1

frequencies = {}
for w in words_df.index:
    frequencies[w] = int(words_df.loc[w])

for key in list(frequencies.keys()):
    if key in names_list:
        frequencies.pop(key)

import nltk

words = ' '.join(frequencies.keys())
is_noun = lambda pos: pos[:2] == 'NN'
tok = nltk.word_tokenize(words)
nouns = [word for (word, pos) in nltk.pos_tag(tok) if is_noun(pos)]

for noun in nouns:
    frequencies.pop(noun)

from nltk.corpus import wordnet as wn
wrong_words = []
words = ' '.join(frequencies.keys())
tok = nltk.word_tokenize(words)
for word in tok:
    if not wn.synsets(word):
        wrong_words.append(word)

len(words)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate_from_frequencies(data)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud, interpolation='bilinear')

show_wordcloud(frequencies)

data.to_csv('data.csv')

