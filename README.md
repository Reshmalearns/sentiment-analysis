Problem statement
Business Context

The prices of the stocks of companies listed under a global exchange are influenced by a variety of factors, with the company’s financial performance, innovations and collaborations, and market sentiment being factors that play a significant role. News and media reports can rapidly affect investor perceptions and, consequently, stock prices in the highly competitive financial industry. With the sheer volume of news and opinions from a wide variety of sources, investors and financial analysts often struggle to stay updated and accurately interpret its impact on the market. As a result, investment firms need sophisticated tools to analyze market sentiment and integrate this information into their investment strategies.

Objective
With an ever-rising number of news articles and opinions, an investment startup aims to leverage artificial intelligence to address the challenge of interpreting stock-related news and its impact on stock prices. They have collected historical daily news for a specific company listed under NASDAQ, along with data on its daily stock price and trade volumes.

Tasked with analyzing the data, developing an AI-driven sentiment analysis system that will automatically process and analyze news articles to gauge market sentiment, and summarizing the news at a weekly level to enhance the accuracy of their stock price predictions and optimize investment strategies. This will empower their financial analysts with actionable insights, leading to more informed investment decisions and improved client outcomes.

Data Dictionary
Date: The date the news was released

News: The content of news articles that could potentially affect the company's stock price

Open: The stock price (in $) at the beginning of the day

High: The highest stock price (in $) reached during the day

Low: The lowest stock price (in $) reached during the day

Close: The adjusted stock price (in $) at the end of the day

Volume: The number of shares traded during the day

Label: The sentiment polarity of the news content

1: Positive

0: Neutral

-1: Negative

Libraries
# Basic data handling
import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Warnings
import warnings
warnings.filterwarnings("ignore")

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
Load dataset
#Mount to drive

from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
#load csv fie from google drive
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/stock_news.csv')
#  read the dataset first 5 and last 5 rows

# Display the first 5 rows of the DataFrame.
df.head()
Date	News	Open	High	Low	Close	Volume	Label
0	2019-01-02	The tech sector experienced a significant dec...	41.740002	42.244999	41.482498	40.246914	130672400	-1
1	2019-01-02	Apple lowered its fiscal Q1 revenue guidance ...	41.740002	42.244999	41.482498	40.246914	130672400	-1
2	2019-01-02	Apple cut its fiscal first quarter revenue fo...	41.740002	42.244999	41.482498	40.246914	130672400	-1
3	2019-01-02	This news article reports that yields on long...	41.740002	42.244999	41.482498	40.246914	130672400	-1
4	2019-01-02	Apple's revenue warning led to a decline in U...	41.740002	42.244999	41.482498	40.246914	130672400	-1
# Display the last 5 rows of the DataFrame
df.tail()
Date	News	Open	High	Low	Close	Volume	Label
344	2019-04-30	Media mogul Oprah Winfrey, known for influenc...	50.764999	50.849998	49.7775	48.70879	186139600	-1
345	2019-04-30	European shares fell on Tuesday, with banks u...	50.764999	50.849998	49.7775	48.70879	186139600	-1
346	2019-04-30	This article reports that the S&P 500 reached...	50.764999	50.849998	49.7775	48.70879	186139600	-1
347	2019-04-30	The Federal Reserve is anticipated to keep in...	50.764999	50.849998	49.7775	48.70879	186139600	-1
348	2019-04-30	In the first quarter, South Korea's Samsung E...	50.764999	50.849998	49.7775	48.70879	186139600	0
Data overview
# shape of the dataset
df.shape
(349, 8)
The dataset has
rows = 349
columns = 8
# data types present
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 349 entries, 0 to 348
Data columns (total 8 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Date    349 non-null    object 
 1   News    349 non-null    object 
 2   Open    349 non-null    float64
 3   High    349 non-null    float64
 4   Low     349 non-null    float64
 5   Close   349 non-null    float64
 6   Volume  349 non-null    int64  
 7   Label   349 non-null    int64  
dtypes: float64(4), int64(2), object(2)
memory usage: 21.9+ KB
# duplicates
df.duplicated().sum()
0
# missing values
df.isnull().sum()
0
Date	0
News	0
Open	0
High	0
Low	0
Close	0
Volume	0
Label	0

dtype: int64
Dataset has no missing and duplicate values
# statistical summary
df.describe().T
count	mean	std	min	25%	50%	75%	max
Open	349.0	4.622923e+01	6.442817e+00	3.756750e+01	4.174000e+01	4.597500e+01	5.070750e+01	6.681750e+01
High	349.0	4.670046e+01	6.507321e+00	3.781750e+01	4.224500e+01	4.602500e+01	5.085000e+01	6.706250e+01
Low	349.0	4.574539e+01	6.391976e+00	3.730500e+01	4.148250e+01	4.564000e+01	4.977750e+01	6.586250e+01
Close	349.0	4.492632e+01	6.398338e+00	3.625413e+01	4.024691e+01	4.459692e+01	4.911079e+01	6.480523e+01
Volume	349.0	1.289482e+08	4.317031e+07	4.544800e+07	1.032720e+08	1.156272e+08	1.511252e+08	2.444392e+08
Label	349.0	-5.444126e-02	7.151192e-01	-1.000000e+00	-1.000000e+00	0.000000e+00	0.000000e+00	1.000000e+00
Exploratory Data Analysis
Univariate data analysis
Label
# Distribution of sentiment labels
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Label',color='skyblue')
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Negative (-1)', 'Neutral (0)', 'Positive (1)'])
plt.show()
No description has been provided for this image
# Percentage of each sentiment
label_counts = df['Label'].value_counts(normalize=True) * 100
print("Percentage distribution of sentiment labels:\n", label_counts)
Percentage distribution of sentiment labels:
 Label
 0    48.710602
-1    28.366762
 1    22.922636
Name: proportion, dtype: float64
The news content had

0 - 48.710602% has neutral effect
-1 - 28.366762% has negative effect
1 - 22.922636% has positive effect
Stock price
# Plot histograms for stock price-related columns
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols].hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Stock Features", fontsize=16)
plt.show()
No description has been provided for this image
# Summary statistics
df[numeric_cols].describe()
Open	High	Low	Close	Volume
count	349.000000	349.000000	349.000000	349.000000	3.490000e+02
mean	46.229233	46.700458	45.745394	44.926317	1.289482e+08
std	6.442817	6.507321	6.391976	6.398338	4.317031e+07
min	37.567501	37.817501	37.305000	36.254131	4.544800e+07
25%	41.740002	42.244999	41.482498	40.246914	1.032720e+08
50%	45.974998	46.025002	45.639999	44.596924	1.156272e+08
75%	50.707500	50.849998	49.777500	49.110790	1.511252e+08
max	66.817497	67.062500	65.862503	64.805229	2.444392e+08
The average opening price is around 
46.23
,
w
i
t
h
a
m
a
x
c
l
o
s
e
o
f
64.81 and a minimum of $36.25

The standard deviation for all price columns is about $6.4

No extreme outliers or zero values

The mean volume is 128 million, with a standard deviation of ~43 million, suggesting a wide range of daily trade activity

Price & Volume
plt.figure(figsize=(12, 6))
cols = ['Open', 'High', 'Low', 'Close', 'Volume']

for i, col in enumerate(cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()
No description has been provided for this image
Outliers detection in volume

plt.figure(figsize=(12, 5))
for i, col in enumerate(cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()
No description has been provided for this image
Open - High - Low - Close Prices

All show a fairly symmetrical distribution around the median

open - Some days had unusually high opening prices
high - Peak prices on certain days
low - Even low prices had outliers
close - Some days closed with much higher prices
There are notable outliers on the higher end

Volume

Has the widest range and the most extreme outliers
Volume Strong Right-skewed
Bivariate data analysis
Open, High, Low, Close, Volume behave across the two classes of Label ( 0 ,-1 and 1) i.e neutral influence and positive influence and negative
# Set up the plot style
sns.set(style="whitegrid")

# List of numerical features to plot against 'Label'
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Create boxplots for each feature vs Label
plt.figure(figsize=(16, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='Label', y=feature, data=df, palette='coolwarm')
    plt.title(f'{feature} vs Label')

plt.tight_layout()
plt.show()
No description has been provided for this image
Open - Median varies slightly across labels
High - Higher highs may indicate price rises
Low - Higher lows hint at upward pressure
Close - Clear difference in medians by label
Volume - No clear separation across labels
Correlation heatmap

import matplotlib.pyplot as plt
import seaborn as sns

# Select only numerical columns for correlation
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Label']
corr_matrix = df[numeric_cols].corr()

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True)

# Title
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Show the plot
plt.tight_layout()
plt.show()
No description has been provided for this image
Label has very low positive correlation with open high low and close and no significant correlation with volume
News feature
numerical features don’t show strong relationships with the Label ,NLP preprocessing of the "News" column – this is likely to contain the strongest signals for predicting stock movement.
Text Cleaning Function
import re
import string
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
True
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
Cleaning on News Column
df['Cleaned_News'] = df['News'].apply(clean_text)
Tokenize and Remove Stopwords
 
!pip install nltk
import nltk
nltk.download('punkt_tab')
Requirement already satisfied: nltk in /usr/local/lib/python3.11/dist-packages (3.9.1)
Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from nltk) (8.1.8)
Requirement already satisfied: joblib in /usr/local/lib/python3.11/dist-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.11/dist-packages (from nltk) (2024.11.6)
Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from nltk) (4.67.1)
[nltk_data] Downloading package punkt_tab to /root/nltk_data...
[nltk_data]   Package punkt_tab is already up-to-date!
True
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

df['Tokens'] = df['Cleaned_News'].apply(tokenize_and_remove_stopwords)
df[['News', 'Cleaned_News', 'Tokens']].head()
# Original News text ,Cleaned version (no punctuation, no digits, lowercase)
# Token list with stopwords removed (ready for vectorization)
News	Cleaned_News	Tokens
0	The tech sector experienced a significant dec...	the tech sector experienced a significant decl...	[tech, sector, experienced, significant, decli...
1	Apple lowered its fiscal Q1 revenue guidance ...	apple lowered its fiscal q revenue guidance to...	[apple, lowered, fiscal, q, revenue, guidance,...
2	Apple cut its fiscal first quarter revenue fo...	apple cut its fiscal first quarter revenue for...	[apple, cut, fiscal, first, quarter, revenue, ...
3	This news article reports that yields on long...	this news article reports that yields on longd...	[news, article, reports, yields, longdated, us...
4	Apple's revenue warning led to a decline in U...	apples revenue warning led to a decline in usd...	[apples, revenue, warning, led, decline, usd, ...
Train-Test Split
from sklearn.model_selection import train_test_split

# Features and target
X = df['Tokens']   # Cleaned and tokenized text
y = df['Label']    # Target

# Split into train, validation, test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val)

# Show split sizes
print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))
Train size: 282
Validation size: 32
Test size: 35
Embedding
Word2Vec

GloVe

Sentence Transforme

# Install compatible versions
!pip install numpy==1.23.5 scipy==1.10.1 gensim==4.3.1 --force-reinstall
Collecting numpy==1.23.5
  Using cached numpy-1.23.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
Collecting scipy==1.10.1
  Using cached scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (58 kB)
Collecting gensim==4.3.1
  Using cached gensim-4.3.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)
Collecting smart-open>=1.8.1 (from gensim==4.3.1)
  Using cached smart_open-7.1.0-py3-none-any.whl.metadata (24 kB)
Collecting wrapt (from smart-open>=1.8.1->gensim==4.3.1)
  Using cached wrapt-1.17.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.4 kB)
Using cached numpy-1.23.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
Using cached scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.1 MB)
Using cached gensim-4.3.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.6 MB)
Using cached smart_open-7.1.0-py3-none-any.whl (61 kB)
Using cached wrapt-1.17.2-cp311-cp311-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (83 kB)
Installing collected packages: wrapt, numpy, smart-open, scipy, gensim
  Attempting uninstall: wrapt
    Found existing installation: wrapt 1.17.2
    Uninstalling wrapt-1.17.2:
      Successfully uninstalled wrapt-1.17.2
  Attempting uninstall: numpy
    Found existing installation: numpy 1.23.5
    Uninstalling numpy-1.23.5:
      Successfully uninstalled numpy-1.23.5
  Attempting uninstall: smart-open
    Found existing installation: smart-open 7.1.0
    Uninstalling smart-open-7.1.0:
      Successfully uninstalled smart-open-7.1.0
  Attempting uninstall: scipy
    Found existing installation: scipy 1.10.1
    Uninstalling scipy-1.10.1:
      Successfully uninstalled scipy-1.10.1
  Attempting uninstall: gensim
    Found existing installation: gensim 4.3.1
    Uninstalling gensim-4.3.1:
      Successfully uninstalled gensim-4.3.1
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
jaxlib 0.5.1 requires numpy>=1.25, but you have numpy 1.23.5 which is incompatible.
jaxlib 0.5.1 requires scipy>=1.11.1, but you have scipy 1.10.1 which is incompatible.
cvxpy 1.6.5 requires scipy>=1.11.0, but you have scipy 1.10.1 which is incompatible.
imbalanced-learn 0.13.0 requires numpy<3,>=1.24.3, but you have numpy 1.23.5 which is incompatible.
tensorflow 2.18.0 requires numpy<2.1.0,>=1.26.0, but you have numpy 1.23.5 which is incompatible.
scikit-image 0.25.2 requires numpy>=1.24, but you have numpy 1.23.5 which is incompatible.
scikit-image 0.25.2 requires scipy>=1.11.4, but you have scipy 1.10.1 which is incompatible.
bigframes 1.42.0 requires numpy>=1.24.0, but you have numpy 1.23.5 which is incompatible.
albumentations 2.0.5 requires numpy>=1.24.4, but you have numpy 1.23.5 which is incompatible.
blosc2 3.3.0 requires numpy>=1.26, but you have numpy 1.23.5 which is incompatible.
jax 0.5.2 requires numpy>=1.25, but you have numpy 1.23.5 which is incompatible.
jax 0.5.2 requires scipy>=1.11.1, but you have scipy 1.10.1 which is incompatible.
albucore 0.0.23 requires numpy>=1.24.4, but you have numpy 1.23.5 which is incompatible.
chex 0.1.89 requires numpy>=1.24.1, but you have numpy 1.23.5 which is incompatible.
treescope 0.1.9 requires numpy>=1.25.2, but you have numpy 1.23.5 which is incompatible.
thinc 8.3.6 requires numpy<3.0.0,>=2.0.0, but you have numpy 1.23.5 which is incompatible.
xarray 2025.1.2 requires numpy>=1.24, but you have numpy 1.23.5 which is incompatible.
pymc 5.21.2 requires numpy>=1.25.0, but you have numpy 1.23.5 which is incompatible.
Successfully installed gensim-4.3.1 numpy-1.23.5 scipy-1.10.1 smart-open-7.1.0 wrapt-1.17.2
from gensim.models import KeyedVectors

# Path to the converted GloVe file
glove_path = '/content/drive/MyDrive/glove.6B.100d.txt.word2vec'

# Load the model (not binary since it's a .txt)
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)

# Check vocabulary size
print("✅ GloVe model loaded! Vocabulary size:", len(glove_model))
✅ GloVe model loaded! Vocabulary size: 400000
import numpy as np

# Function to get average GloVe vector for a sentence
def sentence_to_glove_vector(tokens, model, vector_size=100):
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Apply to all tokenized texts
X_glove = np.array([sentence_to_glove_vector(tokens, glove_model) for tokens in df['Tokens']])

# Check shape
print(" GloVe Embedding Shape:", X_glove.shape)
 GloVe Embedding Shape: (349, 100)
Split GloVe Embeddings into Train, Validation, and Test Sets

# First, reset the index to align everything cleanly
df = df.reset_index(drop=True)

# Re-split on the index
from sklearn.model_selection import train_test_split

# Full index
indices = list(range(len(df)))

# Split into train+val and test
train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42, stratify=df['Label'])

# Split train_val into train and val
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42, stratify=df['Label'].iloc[train_val_indices])

# Final embedding splits
X_train_glove = X_glove[train_indices]
X_val_glove = X_glove[val_indices]
X_test_glove = X_glove[test_indices]

# Corresponding labels
y_train_glove = df['Label'].iloc[train_indices].values
y_val_glove = df['Label'].iloc[val_indices].values
y_test_glove = df['Label'].iloc[test_indices].values

# Confirm
print("Train Embeddings:", X_train_glove.shape)
print("Validation Embeddings:", X_val_glove.shape)
print("Test Embeddings:", X_test_glove.shape)
Train Embeddings: (282, 100)
Validation Embeddings: (32, 100)
Test Embeddings: (35, 100)
 
print("------"*40)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# Train the model
glove_model = LogisticRegression(max_iter=1000, random_state=42)
glove_model.fit(X_train_glove, y_train_glove)

# Predict on validation set
y_val_pred = glove_model.predict(X_val_glove)

# Evaluate
print("Validation Set Evaluation (GloVe + Logistic Regression):")
print(classification_report(y_val_glove, y_val_pred))

f1 = f1_score(y_val_glove, y_val_pred, average='weighted')
print("Weighted F1 Score:", f1)
Validation Set Evaluation (GloVe + Logistic Regression):
              precision    recall  f1-score   support

          -1       0.45      0.56      0.50         9
           0       0.55      0.69      0.61        16
           1       0.00      0.00      0.00         7

    accuracy                           0.50        32
   macro avg       0.33      0.41      0.37        32
weighted avg       0.40      0.50      0.45        32

Weighted F1 Score: 0.4461805555555556
model is doing okay on neutral and negative classes, but really struggling with the positive class (1)
Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga']
}

# Setup grid search
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                           param_grid,
                           scoring='f1_weighted',
                           cv=5,
                           n_jobs=-1)

# Fit
grid_search.fit(X_train_glove, y_train_glove)

# Best model
best_glove_model = grid_search.best_estimator_

# Validation predictions with tuned model
y_val_pred_tuned = best_glove_model.predict(X_val_glove)

# Evaluate
print("Tuned Model - Validation Set Evaluation:")
print(classification_report(y_val_glove, y_val_pred_tuned))
print("Weighted F1 Score (Tuned):", f1_score(y_val_glove, y_val_pred_tuned, average='weighted'))
Tuned Model - Validation Set Evaluation:
              precision    recall  f1-score   support

          -1       0.50      0.56      0.53         9
           0       0.59      0.81      0.68        16
           1       0.00      0.00      0.00         7

    accuracy                           0.56        32
   macro avg       0.36      0.46      0.40        32
weighted avg       0.44      0.56      0.49        32

Weighted F1 Score (Tuned): 0.4901315789473684
tuned Logistic Regression model with GloVe embeddings did improve slightly:
Accuracy: up from 50% → 56%

Weighted F1: up from ~0.45 → ~0.49

Still struggling with class 1 though

Train Word2Vec

from gensim.models import Word2Vec
import numpy as np

# Tokenized column
sentences = df['Tokens']  # this should be a list of lists of tokens

# Train Word2Vec model
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1, seed=42)

# Function to get average Word2Vec embedding for each sentence
def get_avg_w2v_embedding(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# Create Word2Vec embeddings
X_w2v = np.array([get_avg_w2v_embedding(tokens, w2v_model, 100) for tokens in sentences])
TrainML Model (Logistic Regression) using Word2Vec embeddings

# Split Word2Vec embeddings using the same indices
X_train_w2v = X_w2v[train_indices]
X_val_w2v = X_w2v[val_indices]
X_test_w2v = X_w2v[test_indices]

# Corresponding labels
y_train_w2v = df['Label'].iloc[train_indices].values
y_val_w2v = df['Label'].iloc[val_indices].values
y_test_w2v = df['Label'].iloc[test_indices].values

# Confirm
print("Train Embeddings (Word2Vec):", X_train_w2v.shape)
print("Validation Embeddings:", X_val_w2v.shape)
print("Test Embeddings:", X_test_w2v.shape)
Train Embeddings (Word2Vec): (282, 100)
Validation Embeddings: (32, 100)
Test Embeddings: (35, 100)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# Train model
w2v_model_lr = LogisticRegression(max_iter=1000, random_state=42)
w2v_model_lr.fit(X_train_w2v, y_train_w2v)

# Predict on validation set
y_val_pred_w2v = w2v_model_lr.predict(X_val_w2v)

# Evaluate
print("Validation Set Evaluation (Word2Vec + Logistic Regression):")
print(classification_report(y_val_w2v, y_val_pred_w2v))
print("Weighted F1 Score:", f1_score(y_val_w2v, y_val_pred_w2v, average='weighted'))
Validation Set Evaluation (Word2Vec + Logistic Regression):
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00         9
           0       0.50      1.00      0.67        16
           1       0.00      0.00      0.00         7

    accuracy                           0.50        32
   macro avg       0.17      0.33      0.22        32
weighted avg       0.25      0.50      0.33        32

Weighted F1 Score: 0.3333333333333333
The model is heavily biased toward class 0, predicting it for nearly everything
Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga']
}

# Setup grid search
grid_search_w2v = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                               param_grid,
                               scoring='f1_weighted',
                               cv=5,
                               n_jobs=-1)

# Fit
grid_search_w2v.fit(X_train_w2v, y_train_w2v)

# Best model
best_w2v_model = grid_search_w2v.best_estimator_

# Predict on validation
y_val_pred_w2v_tuned = best_w2v_model.predict(X_val_w2v)

# Evaluate
print("Tuned Model - Validation Set Evaluation (Word2Vec):")
print(classification_report(y_val_w2v, y_val_pred_w2v_tuned))
print("Weighted F1 Score (Tuned):", f1_score(y_val_w2v, y_val_pred_w2v_tuned, average='weighted'))
Tuned Model - Validation Set Evaluation (Word2Vec):
              precision    recall  f1-score   support

          -1       0.50      0.11      0.18         9
           0       0.50      0.94      0.65        16
           1       0.00      0.00      0.00         7

    accuracy                           0.50        32
   macro avg       0.33      0.35      0.28        32
weighted avg       0.39      0.50      0.38        32

Weighted F1 Score (Tuned): 0.3772233201581028
Slight improvement in class -1 precision/recall.

Still zero recall for class 1.

Weighted F1-score improved a little from 0.33 → 0.38

Word2Vec embeddings aren't separating the classes well enough

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
lr_model.fit(X_train_glove, y_train)

# Predict on validation set
val_preds_lr = lr_model.predict(X_val_glove)

# Evaluate
print("Logistic Regression - Validation Set Evaluation")
print("Accuracy:", accuracy_score(y_val, val_preds_lr))
print("\nClassification Report:\n", classification_report(y_val, val_preds_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, val_preds_lr))
Logistic Regression - Validation Set Evaluation
Accuracy: 0.5

Classification Report:
               precision    recall  f1-score   support

          -1       0.45      0.56      0.50         9
           0       0.55      0.69      0.61        16
           1       0.00      0.00      0.00         7

    accuracy                           0.50        32
   macro avg       0.33      0.41      0.37        32
weighted avg       0.40      0.50      0.45        32


Confusion Matrix:
 [[ 5  4  0]
 [ 4 11  1]
 [ 2  5  0]]
model predicts neutral (0) sentiment best, but struggles with positive (1)

from gensim.models import Word2Vec

# Train Word2Vec on tokenized text
w2v_model = Word2Vec(
    sentences=df['Tokens'],  # list of tokenized sentences
    vector_size=100,         # embedding size (same as GloVe)
    window=5,                # context window size
    min_count=1,             # include all words
    workers=4,               # number of threads
    sg=1                     # skip-gram (1), CBOW would be 0
)

# Vocabulary size
print("Word2Vec vocabulary size:", len(w2v_model.wv))
Word2Vec vocabulary size: 3359
Convert tokenized sentences into average Word2Vec vectors

import numpy as np

# Function to convert a sentence to an average Word2Vec vector
def sentence_to_w2v_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(vectors, axis=0)

# Apply the function to all tokenized texts
X_w2v = np.array([sentence_to_w2v_vector(tokens, w2v_model) for tokens in df['Tokens']])
print("Word2Vec Embedding Shape:", X_w2v.shape)
Word2Vec Embedding Shape: (349, 100)
Split Word2Vec Embeddings into Train, Validation, and Test Sets

# Use previously split indices to slice X_w2v accordingly
X_train_w2v = X_w2v[X_train.index]
X_val_w2v = X_w2v[X_val.index]
X_test_w2v = X_w2v[X_test.index]

print("Train Embeddings:", X_train_w2v.shape)
print("Validation Embeddings:", X_val_w2v.shape)
print("Test Embeddings:", X_test_w2v.shape)
Train Embeddings: (282, 100)
Validation Embeddings: (32, 100)
Test Embeddings: (35, 100)
Train and Evaluate Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the model
w2v_model = LogisticRegression(max_iter=1000)
w2v_model.fit(X_train_w2v, y_train)

# Predict on validation set
y_val_pred_w2v = w2v_model.predict(X_val_w2v)

# Evaluate
print("Logistic Regression - Word2Vec - Validation Set Evaluation")
print("Accuracy:", accuracy_score(y_val, y_val_pred_w2v))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred_w2v))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred_w2v))
Logistic Regression - Word2Vec - Validation Set Evaluation
Accuracy: 0.5

Classification Report:
               precision    recall  f1-score   support

          -1       0.00      0.00      0.00         9
           0       0.50      1.00      0.67        16
           1       0.00      0.00      0.00         7

    accuracy                           0.50        32
   macro avg       0.17      0.33      0.22        32
weighted avg       0.25      0.50      0.33        32


Confusion Matrix:
 [[ 0  9  0]
 [ 0 16  0]
 [ 0  7  0]]
model only predicted class 0 (Neutral) for all validation Precision and recall are 0 for classes -1 and 1.
