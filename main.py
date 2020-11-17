from os import listdir
from os.path import join, isfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def readInMovies(inputdir, positive):
    filesToRead = [f for f in listdir(inputdir) if isfile(join(inputdir, f))]
    df = pd.DataFrame(columns=['positive', 'text'])
    for file in filesToRead:
        f = open(inputdir + "/" + file, "r")
        if f.mode == 'r':
            words = f.read()

            df = df.append({'positive': positive, 'text': words}, ignore_index=True)
    return df

def readInWords(filename):
    df = pd.DataFrame(columns=['words'])
    f = open(filename, "r")
    if f.mode == 'r':
        words = f.read().lower().split()
        df = df.append({'words': words}, ignore_index=True)
    return df

def lexicon_based_classification(test_set, stemmed_words_pos, stemmed_words_neg):

    # classify documents with lexicon based classifyer
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index, text in test_set.iterrows():
        countPosWords = 0
        countNegWords = 0
        for word in text.iloc[1]:
            if word in stemmed_words_pos:
                countPosWords += 1
            if word in stemmed_words_neg:
                countNegWords += 1
        positive = False
        if countPosWords > countNegWords:
            positive = True
        if text.iloc[0]:
            if positive:
                tp += 1
            else:
                fn += 1
        else:
            if positive:
                fp += 1
            else:
                tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall / (precision + recall))

    print("Results of lexicon based classifier:")
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(f1) + "\n")

if __name__ == "__main__":
    neg_movies_folder = r"D:\GitHub\ETH_NLP\files\moviereviews\neg"
    pos_movies_folder = r"D:\GitHub\ETH_NLP\files\moviereviews\pos"

    neg_words_file = r"D:\GitHub\ETH_NLP\files\opinion-lexicon-English\negative-words.txt"
    pos_words_file = r"D:\GitHub\ETH_NLP\files\opinion-lexicon-English\positive-words.txt"

    # Read in all files
    allMovies = readInMovies(neg_movies_folder, False)
    allMovies = allMovies.append(readInMovies(pos_movies_folder, True)).reset_index(drop=True)

    allWords_pos = readInWords(pos_words_file)
    allWords_neg = readInWords(neg_words_file)
    print("All files read in")

    stopwords = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r"\w+")

    allMoviesPreprocessed = pd.DataFrame(columns=['positive', 'tokens'])
    #Text preprocessing
    print("Text preprocessing in progress...\n")
    for index, row in allMovies.iterrows():
        tokenized_words = tokenizer.tokenize(row['text'])
        lower_script_words = [w.lower() for w in tokenized_words]
        filtered_sentence = [w for w in lower_script_words if w not in stopwords]
        stemmed_words = [ps.stem(w) for w in filtered_sentence]

        allMoviesPreprocessed = allMoviesPreprocessed.append({'positive': allMovies.iloc[index, 0], 'tokens': stemmed_words}, ignore_index=True)
    print("Text preprocessing completed\n")

    # Check if some word of pos/neg list is in stopwords
    print("Words of pos/neg list contained in stopwords:\n")
    for word in allWords_neg.iloc[0, 0]:
        if word in stopwords:
            print(word)
    for word in allWords_pos.iloc[0, 0]:
        if word in stopwords:
            print(word)

    # Pos/neg list word preprocessing (just stemming)
    stemmed_words_pos = [ps.stem(w) for w in allWords_pos.iloc[0, 0]]
    stemmed_words_neg = [ps.stem(w) for w in allWords_neg.iloc[0, 0]]

    # split into training and testset
    shuffled_allMovies = allMoviesPreprocessed.sample(frac=1).reset_index(drop=True)
    training_set = shuffled_allMovies.iloc[0:1600]
    test_set = shuffled_allMovies.iloc[1600:2000]

    lexicon_based_classification(test_set=test_set, stemmed_words_pos=stemmed_words_pos, stemmed_words_neg=stemmed_words_neg)

    # Classification with logistic regression, using BOW

    # Bring data into required shape
    x_train = training_set['tokens']
    y_train = training_set['positive'].astype(int)
    x_train = [" ".join(x) for x in x_train]
    x_test = test_set['tokens']
    x_test = [" ".join(x) for x in x_test]
    y_test = test_set['positive'].astype(int)
    all_words = x_train + x_test  # Needed to create BOW

    # Create BOW
    bow_converter = CountVectorizer()
    word_vector = bow_converter.fit(all_words)
    x_train = word_vector.transform(x_train)
    x_test = word_vector.transform(x_test)

    # Run Logistic Regression
    model = LogisticRegression(C=1.0, max_iter=1000).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Logistic regression results:")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(accuracy))
    f1 = f1_score(y_test, y_pred)
    print("F1: " + str(f1))
