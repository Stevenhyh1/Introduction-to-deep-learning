import numpy as np
from loss import Criterion

class CTCLoss(Criterion):
    """
    CTC loss
    """
    def __init__(self, blank = 0):
        """
        Argument:
            blank (int, optional) – blank label index. Default 0.
        """
        #-------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.target_length = None
        self.input_length = None
        self.alphas = []
        self.betas = []
        self.gammas = []

        self.blank = blank
        #<---------------------------------------------

    def __call__(self, a, b, c, d):
        #-------------------------------------------->
        # Don't Need Modify
        return self.forward(a, b, c, d)
        #<---------------------------------------------
    
    def forward(self, logits, target, input_length, target_length):
        #-------------------------------------------->
        # Don't Need Modify
        self.logits = logits  # (T, B, D)
        self.target = target
        self.input_length = input_length
        self.target_length = target_length

        #<---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths 
        #####  and then the mean over the batch is taken

        B, L = target.shape
        total_loss = np.zeros(B)

        for b in range(B):
            #-------------------------------------------->
            # Computing CTC Loss for single batch
            # Process: 
            #     Extend Sequence with blank -> 
            #     Compute forward probabilities ->
            #     Compute backward probabilities ->
            #     Compute posteriors using total probability function
            #     Compute Cross Entropy lost and take average on batches
            #<---------------------------------------------

            #-------------------------------------------->
            # Your Code goes here
            # import pdb
            # pdb.set_trace()
            cur_logit = logits[:input_length[b], b, :]
            S_ext, Skip_Connect = self._ext_seq_blank(target[b, :target_length[b]])
            alpha = self._forward_prob(cur_logit, S_ext, Skip_Connect)  # (T, N)
            self.alphas.append(alpha)
            beta = self._backward_prob(cur_logit, S_ext, Skip_Connect)  # (T, N)
            self.betas.append(beta)
            gamma = self._post_prob(alpha, beta)  # (T, N)
            self.gammas.append(gamma)
            T, N = gamma.shape
            loss = 0
            for t in range(T):
                for i in range(N):
                    loss += gamma[t, i] * np.log(cur_logit[t, S_ext[i]])
            loss *= -1
            total_loss[b] = loss
            # raise NotImplementedError
            #<---------------------------------------------
        total_loss = np.mean(total_loss)
        return total_loss

    def derivative(self):

        #-------------------------------------------->
        # Don't Need Modify
        L, B, H = self.logits.shape
        dy = np.zeros((L, B, H))
        #<---------------------------------------------

        for b in range(B):
            #-------------------------------------------->
            # Computing CTC Derivative for single batch
            #<---------------------------------------------
            
            #-------------------------------------------->
            # Your Code goes here
            cur_logit = self.logits[:self.input_length[b], b, :]
            S_ext, Skip_Connect = self._ext_seq_blank(self.target[b, :self.target_length[b]])
            T, N = self.gammas[b].shape
            for t in range(T):
                for i in range(N):
                    dy[t, b, S_ext[i]] -= self.gammas[b][t, i] / cur_logit[t, S_ext[i]]
            #<---------------------------------------------
        
        return dy

    def _ext_seq_blank(self, target):
        """
        Argument:
            target (np.array, dim = 1) - target output
        Return:
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Connect (np.array, dim = 1) - skip connections
        """
        S_ext = []
        Skip_Connect = []

        #-------------------------------------------->

        # Your Code goes here
        # import pdb; pdb.set_trace()
        for i, label in enumerate(target):

            S_ext.append(0)
            Skip_Connect.append(0)

            S_ext.append(label)
            if i > 0 and label != target[i-1]:
                Skip_Connect.append(1)
            else:
                Skip_Connect.append(0)

        S_ext.append(0)
        Skip_Connect.append(0)
        # raise NotImplementedError
        #<---------------------------------------------

        return S_ext, Skip_Connect

    def _forward_prob(self, logits, S_ext, Skip_Conn):
        """
        Argument:
            logits (np.array, dim = (input_len, channel)) - predict probabilities
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Conn
        Return:
            alpha (np.array, dim = (output len, out channel)) - forward probabilities
        """
        N, T = len(S_ext), len(logits)
        alpha = np.zeros(shape=(T, N))

        #-------------------------------------------->
        
        # Your Code goes here

        alpha[0, 0] = logits[0, S_ext[0]]
        alpha[0, 1] = logits[0, S_ext[1]]
        for t in range(1, T):
            alpha[t, 0] = alpha[t-1, 0] * logits[t, S_ext[0]]
            for i in range(1, N):
                alpha[t, i] = alpha[t-1, i-1] + alpha[t-1, i]
                if Skip_Conn[i]:
                    alpha[t, i] += alpha[t-1, i-2]
                alpha[t, i] *= logits[t, S_ext[i]]
        #<---------------------------------------------

        return alpha

    def _backward_prob(self, logits, S_ext, Skip_Conn):
        """
        Argument:
            logits (np.array, dim = (input len, channel)) - predict probabilities
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Conn - 
        Return:
            beta (np.array, dim = (output len, out channel)) - backward probabilities
        """
        N, T = len(S_ext), len(logits)
        beta = np.zeros(shape=(T, N))

        #-------------------------------------------->
        # Your Code goes here
        beta[T-1, N-1] = 1
        beta[T-1, N-2] = 1
        for t in range(T-2, -1, -1):
            beta[t, N-1] = beta[t+1, N-1] * logits[t+1, S_ext[N-1]]
            for i in range(N-2, -1, -1):
                beta[t, i] = beta[t+1, i] * logits[t+1, S_ext[i]] + beta[t+1, i+1] * logits[t+1, S_ext[i+1]]
                if i <= N-3 and Skip_Conn[i+2]:
                    beta[t, i] += beta[t+1, i+2] * logits[t+1, S_ext[i+2]]
        # raise NotImplementedError
        #<---------------------------------------------

        return beta


    def _post_prob(self, alpha, beta):
        """
        Argument:
            alpha (np.array) - forward probability
            beta (np.array) - backward probability
        Return:
            gamma (np.array) - posterior probability
        """
        #-------------------------------------------->
        
        # Your Code goes here
        assert alpha.shape == beta.shape, 'The shape of forward probability and backward probability not match'
        # import pdb; pdb.set_trace()
        T, N = alpha.shape
        gamma = np.zeros(shape=(T, N))
        for t in range(T):
            sumgamma = 0
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                sumgamma += gamma[t, i]
            for i in range(N):
                gamma[t, i] /= sumgamma
        #<---------------------------------------------

        return gamma
