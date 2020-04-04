import torch

def collate_wrapper(batch_data):

    inputs, targets= zip(*batch_data)
    
    batch_size = len(inputs)
    feature_dim = inputs[0].shape[-1]
    
    inputs = list(inputs)
    input_lens = [len(input) for input in inputs]  
    longest_input = max(input_lens)
    padded_input = torch.zeros(batch_size, longest_input, feature_dim)
    for i, input_len in enumerate(input_lens):
        cur_input = inputs[i]
        padded_input[i,0:input_len] = cur_input
    padded_input = padded_input.permute(1,0,2)

    targets = list(targets)
    target_lens = [len(label) for label in targets]
    longest_target = max(target_lens)
    padded_target = torch.zeros(batch_size, longest_target)
    for i, target_len in enumerate(target_lens):
        cur_target = targets[i]
        padded_target[i,0:target_len] = cur_target
    
    return padded_input, padded_target, input_lens, target_lens

def collate_wrapper_test(batch_data):

    inputs = batch_data
    # import pdb; pdb.set_trace()

    batch_size = len(inputs)
    feature_dim = inputs[0].shape[-1]
    
    inputs = list(inputs)
    input_lens = [len(input) for input in inputs]  
    longest_input = max(input_lens)
    padded_input = torch.zeros(batch_size, longest_input, feature_dim)
    for i, input_len in enumerate(input_lens):
        cur_input = inputs[i]
        padded_input[i,0:input_len] = cur_input
    padded_input = padded_input.permute(1,0,2)
    
    return padded_input, input_lens,