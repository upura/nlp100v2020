import random
from torchtext import data
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(
            emb_dim,
            enc_hid_dim,
            bidirectional=True
        )
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(
            torch.cat((
                hidden[-2, :, :],
                hidden[-1, :, :]),
                dim=1
            )
        ))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(
            torch.cat((
                repeated_decoder_hidden,
                encoder_outputs
            ), dim=2)
        ))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, encoder_outputs):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden,
            encoder_outputs
        )

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(
            torch.cat((
                output,
                weighted_encoder_rep,
                embedded
            ), dim=1)
        )

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            max_len,
            batch_size,
            trg_vocab_size
        )

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(
                output,
                hidden,
                encoder_outputs
            )
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, iterator):
    model.eval()
    y_preds = []
    with torch.no_grad():

        for _, batch in enumerate(iterator):

            ja = batch.JA
            en = batch.EN

            ja = ja.to(device)
            en = en.to(device)

            output = model(ja, en, 0)     # turn off teacher forcing
            output = output.to(device)
            y_pred = torch.max(output, 2)[1]
            y_preds.append(y_pred)

    return y_preds


def decode_text(en):
    en = list(en)
    en = [EN.vocab.itos[e] for e in en]
    text = ' '.join(en[1:])
    return text


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_model(JA, EN, device):
    INPUT_DIM = len(JA.vocab)     # 入力データの単語数
    OUTPUT_DIM = len(EN.vocab)    # 出力データの単語数

    ENC_EMB_DIM = 32               # Encoder用のembeddingの次元数
    DEC_EMB_DIM = 32               # Decoder用のembeddingの次元数
    ENC_HID_DIM = 64               # Encoder用の隠れ層の次元数
    DEC_HID_DIM = 64               # Decoder用の隠れ層の次元数
    ATTN_DIM = 8                   # Attentionの隠れ層の次元数
    ENC_DROPOUT = 0.5              # Encoder用のDropout確率
    DEC_DROPOUT = 0.5              # Decoder用のDropout確率

    enc = Encoder(
        input_dim=INPUT_DIM,
        emb_dim=ENC_EMB_DIM,
        enc_hid_dim=ENC_HID_DIM,
        dec_hid_dim=DEC_HID_DIM,
        dropout=ENC_DROPOUT
    )

    attn = Attention(
        enc_hid_dim=ENC_HID_DIM,
        dec_hid_dim=DEC_HID_DIM,
        attn_dim=ATTN_DIM
    )

    dec = Decoder(
        output_dim=OUTPUT_DIM,
        emb_dim=DEC_EMB_DIM,
        enc_hid_dim=ENC_HID_DIM,
        dec_hid_dim=DEC_HID_DIM,
        dropout=DEC_DROPOUT,
        attention=attn
    )

    model = Seq2Seq(
        encoder=enc,
        decoder=dec
    )

    model.apply(init_weights)
    model = model.to(device)
    model.load_state_dict(torch.load('ch10/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


EN = data.Field(sequential=True, init_token='<sos>', eos_token='<eos>', lower=True)
JA = data.Field(sequential=True, init_token='<sos>', eos_token='<eos>', lower=True)

use_small = True
if use_small:
    train_path = 'kyoto-train_small.txt'
    test_path = 'kyoto-test_small.txt'
else:
    train_path = 'kyoto-train.txt'
    test_path = 'kyoto-test.txt'

train_data, test_data = data.TabularDataset.splits(
    path='ch10/kftt-data-1.0/data/tok',
    train=train_path,
    test=test_path,
    format='tsv',
    fields=[('EN', EN), ('JA', JA)])

EN.build_vocab(train_data, min_freq=2)
JA.build_vocab(train_data, min_freq=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), device=device, batch_size=128, sort=False)

model = load_model(JA, EN, device)
y_preds = predict(model, test_iterator)

batch = next(iter(test_iterator))
references_corpus = (decode_text(torch.transpose(batch.EN, 0, 1)[1])).split()
candidate_corpus = decode_text(torch.transpose(y_preds[0], 0, 1)[1]).split()
print(bleu_score(candidate_corpus, references_corpus))
