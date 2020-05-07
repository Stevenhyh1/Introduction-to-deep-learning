import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)

    N = y_probs.shape[0]
    T = y_probs.shape[1]
    y_probs = y_probs.reshape(N,T)
    forward_prob = 1.0
    s = ""
    for t in range(T):
        cur_prob =  y_probs[:,t]
        cur_idx = np.argmax(cur_prob,axis=0)
        forward_prob *= np.max(cur_prob,axis=0)
        if cur_idx == 0:
            s += '-'
        else:
            s += SymbolSets[cur_idx-1]
    forward_path = s[0] if s[0] != '-' else ""
    for i in range(T-1):
        i += 1
        if s[i] != '-' and s[i] != s[i-1]:
           forward_path += s[i]

    #Viterbi
    # BP = np.zeros((N,T,B))
    # BP[0,0,:]=-1
    # Bscr = -np.ones((N,T,B))
    # Bscr[0][0] = y_probs[0][0]
    # for t in range(T-1):
    #     t += 1
    #     BP[0,t,:] = 1
    #     Bscr[0,t,:] = Bscr[0,t-1,:]*y_probs[0,t,:]
    #     for i in range(N-1):
    #         i += 1
    #         BP[i,t,:][np.where(Bscr[i,t-1,:]>Bscr[i-1,t-1,:])] = i
    #         BP[i,t,:][np.where(Bscr[i,t-1,:]<=Bscr[i-1,t-1,:])] = i-1
    #         Bscr[i,t,:][np.where(Bscr[i,t-1,:]>Bscr[i-1,t-1,:])] = Bscr[i,t-1,:]*y_probs[i,t,:]
    #         Bscr[i,t,:][np.where(Bscr[i,t-1,:]<=Bscr[i-1,t-1,:])] = Bscr[i-1,t-1,:]*y_probs[i,t,:]
    # import pdb; pdb.set_trace()


    return (forward_path, forward_prob)
    # raise NotImplementedError



##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''

def InitializePaths(SymbolSets, y):
    
    InitialBlankPathScore = {}
    InitialPathScore = {}

    path = ""
    InitialBlankPathScore[path] = y[0]
    InitialPathsWithFinalBlank = {path}

    InitialPathsWithFinalSymbol = set()
    for c in range(len(SymbolSets)):
        path = SymbolSets[c]
        InitialPathScore[path] = y[c+1]
        InitialPathsWithFinalSymbol.add(path)
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):

    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    scorelist = []
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    
    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])
    
    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth-1] if BeamWidth < len(scorelist) else scorelist[-1]

    PrunedPathsWithTerminalBlank = set()
    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]
    
    PrunedPathsWithTerminalSymbol = set()
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]
    
    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, y):

    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {}

    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.add(path)
        UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

    for path in PathsWithTerminalSymbol:
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path]*y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = PathScore[path]*y[0]
    
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore,PathScore, SymbolSet, y):
    UpdatedPathsWithTerminalSymbol = set()
    UpdatedPathScore = {}

    for path in PathsWithTerminalBlank:
        for i in range(len(SymbolSet)):
            c = SymbolSet[i]
            newpath = path + c
            UpdatedPathsWithTerminalSymbol.add(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path]*y[i+1]
    
    for path in PathsWithTerminalSymbol:
        for i in range(len(SymbolSet)):
            c = SymbolSet[i]
            newpath = path if (c==path[-1]) else path+c
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
    
    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol,BlankPathScore,PathScore):

    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPaths, FinalPathScore

def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    PathScore = {}
    BlankPathScore = {}
    N = y_probs.shape[0]
    T = y_probs.shape[1]
    y_probs = y_probs.reshape(N,T) 
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0])
    for t in range(T-1):
        t += 1
        PathsWithTerminalBlank, PathsWithTerminalSymbol,BlankPathScore,PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,NewBlankPathScore, NewPathScore, BeamWidth)
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,PathsWithTerminalSymbol,BlankPathScore,PathScore,y_probs[:,t])
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank,PathsWithTerminalSymbol,BlankPathScore,PathScore,SymbolSets,y_probs[:,t])                                                                          

    MergedPaths, mergedPathScores = MergeIdenticalPaths(NewPathsWithTerminalBlank,NewPathsWithTerminalSymbol,NewBlankPathScore,NewPathScore)
    bestscore = -1
    bestPath = ""
    for path in MergedPaths:
        if mergedPathScores[path] > bestscore:
            bestscore = mergedPathScores[path]
            bestPath = path

    return (bestPath, mergedPathScores)
    # cbacbc
    # {'cbacbc': array([0.00195661]), 'cbacbcb': array([0.00061892]), 'cbacbb': array([0.00108495]), 
    # 'cbacbca': array([8.45496711e-05]), 'cbacba': array([0.00014821]), 'cbacb': array([0.00110791])}
    # raise NotImplementedError




