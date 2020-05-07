import sys

sys.path.append('autograder')
from helpers import *

from test_ctc_loss import *
from test_rnn_bptt import *

tests = [
    {
        'name': 'Section 1.1 - Extend Sequence with Blank',
        'autolab': 'Extend Sequence with Blank',
        'handler': test_ctc_extend_seq,
        'value': 1,
    },
    {
        'name': 'Section 1.2 - Posterior Probability',
        'autolab': 'Posterior Probability',
        'handler': test_ctc_posterior_prob,
        'value': 3,
    },
    {
        'name': 'Section 1.3 - CTC Forward',
        'autolab': 'CTC Forward',
        'handler': test_ctc_forward,
        'value': 1,
    },
    {
        'name': 'Section 1.4 - CTC Backward',
        'autolab': 'CTC Backward',
        'handler': test_ctc_backward,
        'value': 1,
    },
    {
        'name': 'Section 2.1 - RNN Seq-to-Seq Forward',
        'autolab': 'RNN Seq-to-Seq Forward',
        'handler': test_rnn_bptt_fwd,
        'value': 2,
    },
    {
        'name': 'Section 2.2 - RNN Seq-to-Seq Backward',
        'autolab': 'RNN Seq-to-Seq Backward',
        'handler': test_rnn_bptt_bwd,
        'value': 2,
    }
]

if __name__ == '__main__':
    run_tests(tests)
