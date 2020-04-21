import torch
import time
from Levenshtein import distance as levenshtein_distance


def train(model, train_loader, criterion, optimizer, epoch, device, writer):
    print(f'Epoch {epoch} starts:')
    train_start = time.time()
    model.train()
    model.to(device)

    perplexity = 0
    epoch_loss = 0
    count = 0

    for padded_input, padded_target, input_lens, target_lens in train_loader:

        with torch.autograd.set_detect_anomaly(True):

            optimizer.zero_grad()
            batch_size = len(input_lens)
            vocab_size = model.vocab_size
            max_len = max(target_lens)
            padded_input = padded_input.to(device)
            padded_target = padded_target.type(torch.LongTensor).to(device)

            predictions = model(padded_input, input_lens, padded_target, istrain=True)  # (batch, max_len, output_dim)
            mask = torch.arange(max_len).unsqueeze(0) >= torch.tensor(target_lens).unsqueeze(1)
            mask = mask.reshape(batch_size*max_len)
            mask = mask.to(device)
            predictions = predictions.reshape(batch_size * max_len, vocab_size)
            padded_target = padded_target.reshape(batch_size*max_len)

            loss = criterion(predictions, padded_target)
            masked_loss = loss.masked_fill_(mask, 0)
            masked_loss = torch.sum(masked_loss)/batch_size
            epoch_loss += masked_loss.detach()
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            perplexity += torch.exp(masked_loss.detach())
            count += 1

    train_end = time.time()
    train_loss = epoch_loss/count
    train_perplexity = perplexity/count
    writer.add_scalar('Loss/Train', train_loss, epoch)
    print(f"Epoch {epoch} completed in: {train_end - train_start}s \t "
          f"Loss: {train_loss} \t Perplexity: {train_perplexity}")


def val(model, val_loader, criterion, epoch, device, writer, index2letter):
    model.eval()
    model.to(device)

    epoch_loss = 0
    perplexity = 0
    count = 0
    distance = 0

    for padded_input, padded_target, input_lens, target_lens in val_loader:
        with torch.no_grad():
            batch_size = len(input_lens)
            vocab_size = model.vocab_size
            max_len = max(target_lens)
            padded_input = padded_input.to(device)
            padded_target = padded_target.type(torch.LongTensor).to(device)

            predictions = model(padded_input, input_lens, padded_target, istrain=True)  # (batch, max_len, vocab)
            mask = torch.arange(max_len).unsqueeze(0) >= torch.tensor(target_lens).unsqueeze(1)
            mask = mask.reshape(batch_size * max_len)
            mask = mask.to(device)
            predictions = predictions.reshape(batch_size * max_len, vocab_size)
            padded_target_loss = padded_target.reshape(batch_size * max_len)

            loss = criterion(predictions, padded_target_loss)
            masked_loss = loss.masked_fill_(mask, 0)
            masked_loss = torch.sum(masked_loss) / batch_size
            epoch_loss += masked_loss.detach()
            perplexity += torch.exp(masked_loss.detach())
            count += 1

            inference = model(padded_input, input_lens, istrain=False)  # (batch, 250, 1)
            cur_dis = 0
            inf_maxlen = inference.size(1)
            # import pdb
            # pdb.set_trace()
            for i in range(batch_size):
                cur_infer = inference[i]
                for j in range(inf_maxlen):
                    if cur_infer[j] == 34 or j == inf_maxlen-1:
                        cur_len = j
                        # import pdb
                        # pdb.set_trace()
                        inf_article = ''
                        for k in range(cur_len+1):
                            inf_article += index2letter[cur_infer[k].item()]
                        target_article = ''
                        for k in range(target_lens[i]):
                            target_article += index2letter[padded_target[i][k].item()]
                        # import pdb
                        # pdb.set_trace()
                        cur_dis += levenshtein_distance(inf_article, target_article)
            # import pdb
            # pdb.set_trace()
            cur_dis /= batch_size
            distance += cur_dis

    val_loss = epoch_loss / count
    val_perplexity = perplexity / count
    val_distance = distance / count
    writer.add_scalar('Loss', val_loss, epoch)
    writer.add_scalar('Perplexity', val_perplexity, epoch)
    writer.add_scalar('Distance', val_distance, epoch)
    print(f"Loss: {val_loss} \t Perplexity: {val_perplexity} \t Distance: {val_distance}")

