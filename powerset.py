from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

array = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

acc_lex = [0.66, 0.6225, 0.6975, 0.6575, 0.685, 0.65, 0.65, 0.65, 0.665, 0.67]
acc_log = [0.86, 0.8575, 0.86, 0.8475, 0.8625, 0.86, 0.835, 0.8775, 0.8325, 0.8575]

abs_mu_diff = 0.19025

permutations = list(powerset(array))
print(len(permutations))
n = 0
for permutation in permutations:
    curSum = 0.0
    for nr in range(0, 10):
        if permutation.__contains__(nr):
            curSum += (acc_lex[nr] - acc_log[nr])
        else:
            curSum += (acc_log[nr] - acc_lex[nr])
    frac = curSum / 10
    if abs(frac) >= abs_mu_diff:
        n = n + 1
print(n)
frac = n / (pow(2, 10))
print(frac)