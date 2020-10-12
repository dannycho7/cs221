"""
Assume MDP of one action so policy is fixed
"""
states = [-2, -1, 0, 1, 2]
V = [0, 0, 0, 0, 0]

def Reward(s , sp):
    if sp == 2:
        return 10
    elif sp == -2:
        return 1.5
    else:
        return 0

def T(s, sp):
    if sp == s - 1:
        return 0.7
    elif sp == s + 1:
        return 0.3
    else:
        return 0

for t in range(100):
    Vp = [0, 0, 0, 0, 0]
    for state1I, state1 in enumerate(states):
        Vp[state1I] = sum(T(state1, state2) * (Reward(state1, state2) + V[state2I]) for state2I, state2 in enumerate(states))
    Vp[0] = 0
    Vp[4] = 0
    V = Vp
    print(V)