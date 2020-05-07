import json

import numpy as np

def assertions(user_vals, expected_vals, test_type, test_name, rtol=1e-5, atol=1e-8):
    if test_type == 'type':
        try:
            assert type(user_vals) == type(expected_vals)
        except Exception as e:
            print('Type error, your type doesnt match the expected type.')
            print('Wrong type for %s' % test_name)
            print('Your type:   ', type(user_vals))
            print('Expected type:', type(expected_vals))
            return False
    elif test_type == 'shape':
        try:
            assert user_vals.shape == expected_vals.shape
        except Exception as e:
            print('Shape error, your shapes doesnt match the expected shape.')
            print('Wrong shape for %s' % test_name)
            print('Your shape:    ', user_vals.shape)
            print('Expected shape:', expected_vals.shape)
            return False
    elif test_type == 'closeness':
        try:
            assert np.allclose(user_vals, expected_vals, rtol=rtol, atol=atol)
        except Exception as e:
            print('Closeness error, your values dont match the expected values.')
            print('Wrong values for %s' % test_name)
            print('Your values:    ', user_vals)
            print('Expected values:', expected_vals)
            return False
    return True

def print_failure(cur_test):
    print('*'*77)
    print('The local autograder will not work if you do not pass %s.' % cur_test)
    print('*'*77)
    print(' ')

def print_name(cur_question):
    print('-'*20)
    print(cur_question)

def print_outcome(short, outcome):
    print(short + ': ', 'PASS' if outcome else '*** FAIL ***')
    print('-'*20)
    print()

def run_tests(tests):
    scores = {}
    for t in tests:
        print_name(t['name'])
        res = t['handler']()
        print_outcome(t['autolab'], res)
        scores[t['autolab']] = t['value'] if res else 0

    print(json.dumps({'scores': scores}))
