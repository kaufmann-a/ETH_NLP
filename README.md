# ETH_NLP
## The projects in NLP Course at ETH in Fall Semester 2020

Logistic Regression results(No regularization):
Tfidf-vectorizer,
clf = LogisticRegression(penalty='none', C=1.0, class_weight='balanced', solver='lbfgs', multi_class='ovr', max_iter=10000)

size=100:
Accuracy: 0.2711864406779661
F1: 0.2970802754520186

size=1000:
Accuracy: 0.502502017756255
F1: 0.5232930064562594

size=10000:
Accuracy: 0.5491525423728814
F1: 0.5431897856817276

size=fullvocabulary:
Accuracy: 0.5512510088781275
F1: 0.5427562392140975

Logistic Regression results(with regularization):
Tfidf-vectorizer
clf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', multi_class='ovr', max_iter=10000)

size=100:
Accuracy: 0.27231638418079096
F1: 0.29314185388131786

size=1000:
Accuracy: 0.537046004842615
F1: 0.5618779655744328

size=10000
Accuracy: 0.6613397901533494
F1: 0.6652409639443237

size=fullvocabulary
Accuracy: 0.661501210653753
F1: 0.6648278834517594

Neural Net without Regularization:
size=100
Accuracy: 0.3672316384180791
F1: 0.3224034361143585

size=1000
Accuracy: 0.5449556093623891
F1: 0.536910558578231

size=10000
Accuracy: 0.5811138014527845
F1: 0.5945056191773649

size=fullsize
Accuracy: 0.5864406779661017
F1: 0.5988715475677137

Neural net with Regularization (kernel-l2 regularization, bias-l2 regularization, several dropout-layers)
size=100
Accuracy: 0.22421307506053267
F1: 0.13126027832794934

size=1000
Accuracy: 0.2891041162227603
F1: 0.21028541603710166

size=10000
Accuracy: 0.2288942695722357
F1: 0.1566140180701775

size=fullsize
Accuracy: 0.22195318805488298
F1: 0.11963278377745913
