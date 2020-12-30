import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from operator import itemgetter

from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Read csv
    data = pd.read_csv("../files/task_b/boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv", index_col=0, header=0, encoding="latin-1")
    data = data.dropna().reset_index(drop=True)
    print ("Finished reading data")

    x = data['title']
    y = data['majortopic']

    stopwords = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r"\w+")

    x_prep = pd.Series(dtype=str)
    # Feature engineering
    print("Feature engineering in progress...\n")
    for index, row in x.iteritems():
        tokenized_words = tokenizer.tokenize(row)
        lower_script_words = [w.lower() for w in tokenized_words]
        filtered_sentence = [w for w in lower_script_words if w not in stopwords]
        stemmed_words = [ps.stem(w) for w in filtered_sentence]
        rem_single_char = [w for w in stemmed_words if len(w) > 1]
        to_string = pd.Series(" ".join(rem_single_char))
        x_prep = x_prep.append(to_append=to_string, ignore_index=True)

    words = {}
    for index, sentence in x_prep.iteritems():
        for word in sentence.split():
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

    # Perform train/validation/test split
    x_train, x_test, y_train, y_test = train_test_split(x_prep, y, test_size=0.2, random_state=None)

    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    one_hot_fitted = LabelBinarizer().fit(y_train.append(y_test))
    y_train_hot = one_hot_fitted.transform(y_train)
    y_test_hot = one_hot_fitted.transform(y_test)
    num_classes = y_test_hot.shape[1]

    # Do prediction with different dictionary sizes
    accuracies_nn = []
    f1_scores_nn = []
    accuracies_lr = []
    f1_scores_lr = []
    labels_plots = []
    for power in range(2, 6):
        size = pow(10, power)
        labels_plots.append(size)
        top_words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}
        top_words = list(top_words.keys())[:size]

        useBOW = False
        useTFIDF = True
        if useBOW:
            # Create BOW
            bow_converter = CountVectorizer()
            word_vector = bow_converter.fit(top_words)
            # print("Vocabulary:")
            # print(word_vector.vocabulary_)
            x_train_vec = word_vector.transform(x_train).toarray()
            x_test_vec = word_vector.transform(x_test).toarray()

            input_shape = (x_train_vec.shape[1],)
            print(f'Feature shape: {input_shape}')
        elif useTFIDF:
            vectorizer = TfidfVectorizer(vocabulary=top_words)
            vectorizer.fit(x_train)
            x_train_vec = vectorizer.transform(x_train).toarray()
            x_test_vec = vectorizer.transform(x_test).toarray()

        NeuralNet = True
        logreg = False
        regularization = True
        if NeuralNet:
            input_shape = (x_train_vec.shape[1],)
            print(f'Feature shape: {input_shape}')

            regul = 1e-5
            model = Sequential()
            if regularization:
                model.add(Input(shape=input_shape))
                model.add(Dense(10000, activation='relu', bias_regularizer=regularizers.l2(regul), kernel_regularizer=regularizers.l2(regul)))
                model.add(Dropout(0.5))
                # model.add(Dense(7500, activation='relu'))
                # model.add(Dropout(0.5))
                model.add(Dense(5000, activation='relu',bias_regularizer=regularizers.l2(regul), kernel_regularizer=regularizers.l2(regul)))
                model.add(Dropout(0.5))
                # model.add(Dense(2500, activation='relu'))
                # model.add(Dropout(0.5))
                model.add(Dense(1000, activation='relu', bias_regularizer=regularizers.l2(regul), kernel_regularizer=regularizers.l2(regul)))
                model.add(Dropout(0.5))
                model.add(Dense(100, activation='relu', bias_regularizer=regularizers.l2(regul), kernel_regularizer=regularizers.l2(regul)))
                model.add(Dense(num_classes, activation='softmax'))
            else:
                model.add(Input(shape=input_shape))
                model.add(Dense(10000, activation='relu'))
                model.add(Dropout(0.5))
                # model.add(Dense(7500, activation='relu'))
                # model.add(Dropout(0.5))
                model.add(Dense(5000, activation='relu'))
                model.add(Dropout(0.5))
                # model.add(Dense(2500, activation='relu'))
                # model.add(Dropout(0.5))
                model.add(Dense(1000, activation='relu' ))
                model.add(Dropout(0.5))
                model.add(Dense(100, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))


            # Compile and train
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Model created:")
            print(model.summary())
            earlystopping = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
            history = model.fit(x_train_vec, y_train_hot, epochs=10, batch_size=100, verbose=1, validation_split=0.25)

            #Test the model
            test_results = model.evaluate(x_test_vec, y_test_hot, verbose=1)
            print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
            plot_history(history=history)

            y_pred = model.predict(x_test_vec)
            y_true_labels = np.array([])
            for row_ind, row in enumerate(y_test_hot):
                for col_ind, column in enumerate(row):
                    if column == 1:
                        y_true_labels = np.append(y_true_labels, col_ind)
            y_true_labels = y_true_labels.astype(int)
            y_pred_labels = np.array([])
            for row_ind, row in enumerate(y_pred):
                curBest = 0
                curBestColnr = 0
                for col_ind, col in enumerate(row):
                    if col > curBest:
                        curBest = col
                        curBestColnr = col_ind
                y_pred_labels = np.append(y_pred_labels, curBestColnr)
            y_pred_labels = y_pred_labels.astype(int)
            print("MLP results:")
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            accuracies_nn.append(accuracy)
            print("Accuracy: " + str(accuracy))
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
            f1_scores_nn.append(f1)
            print("F1: " + str(f1))
        if logreg:
            if regularization:
                clf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', multi_class='ovr', max_iter=10000)
            else:
                clf = LogisticRegression(penalty='none', C=1.0, class_weight='balanced', solver='lbfgs',
                                         multi_class='ovr', max_iter=10000)

            clf.fit(x_train_vec, y_train)
            y_pred = clf.predict(x_test_vec)
            print("Logistic Regression results:")
            accuracy = accuracy_score(y_test, y_pred)
            accuracies_lr.append(accuracy)
            print("Accuracy: " + str(accuracy))
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores_lr.append(f1)
            print("F1: " + str(f1))
    if len(accuracies_nn)>0:
        plt.plot(labels_plots, accuracies_nn, label='accuracies NeuralNet')
    if len(f1_scores_nn)>0:
        plt.plot(labels_plots, f1_scores_nn, label='f1 score NeuralNet')
    if len(accuracies_lr)>0:
        plt.plot(labels_plots, accuracies_lr, label='accuracies Log.Regression')
    if len(f1_scores_lr)>0:
        plt.plot(labels_plots, f1_scores_lr, label='f1 score Log.Regression')
    plt.grid()
    plt.show()

