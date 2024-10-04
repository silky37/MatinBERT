import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForTokenClassification

from torchcrf import CRF


model_revision = 'main'


class BIO_Tag_CRF(CRF):
    def __init__(self, num_tags: int, device, batch_first: bool = False):
        super(BIO_Tag_CRF, self).__init__(num_tags=num_tags, batch_first=batch_first)
        self.device = device
        start_transitions = self.start_transitions.clone().detach()
        transitions = self.transitions.clone().detach()
        assert num_tags % 2 == 1
        num_uniq_labels = (num_tags - 1) // 2
        for i in range(num_uniq_labels, 2 * num_uniq_labels):
            start_transitions[i] = -10000
            for j in range(0, num_tags):
                if j == i or j + num_uniq_labels == i: continue
                transitions[j, i] = -10000
        self.start_transitions = nn.Parameter(start_transitions)
        self.transitions = nn.Parameter(transitions)

    def forward(self, logits, labels, masks):
        
        new_logits, new_labels, new_attention_mask = [], [], []
        for logit, label, mask in zip(logits, labels, masks):
            new_logits.append(logit[mask])
            new_labels.append(label[mask])
            new_attention_mask.append(torch.ones(new_labels[-1].shape[0], dtype=torch.uint8, device=self.device))
        
        padded_logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        loss = -super(BIO_Tag_CRF, self).forward(padded_logits, padded_labels, mask=padded_attention_mask, reduction='mean')
        
        if self.training:
            return (loss, )
        else:
            out = self.decode(padded_logits, mask=padded_attention_mask)
            assert(len(out) == len(labels))
            out_logits = torch.zeros_like(logits)
            for i in range(len(out)):
                k = 0
                for j in range(len(labels[i])):
                    if labels[i][j] == -100: continue
                    out_logits[i][j][out[i][k]] = 1.0
                    k += 1
                assert(k == len(out[i]))
            return (loss, out_logits, )


class BERT_CRF(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        assert('labels' in inputs)
        logits = self.encoder(**inputs)[1]
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(logits, labels, masks)


class BERT_CRF_Contrastive(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF_Contrastive, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        outputs = self.encoder(**inputs, output_hidden_states=True)         # outputs: (loss, logits, hidden_states)
        if 'labels' in inputs:
            logits = outputs[1]     # (B, N)
            last_hidden_states = outputs[2][-1]     # (B, N, D)
            labels = inputs['labels']
            masks = (labels != -100)
            hidden = torch.mean(last_hidden_states, 1)      # (B, D)        # look whole hidden states of all tokens in the sequences
            return self.crf(logits, labels, masks), hidden
        else:
            last_hidden_states = outputs[1][-1]
            hidden = torch.mean(last_hidden_states, 1)
            return None, hidden


class BERT_CRF_WordContrastive(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF_WordContrastive, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config,
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        if "inp_emb_idx" in inputs:
            emb_ids = inputs["inp_emb_idx"]     # (B, # of different words)
            inputs.pop("inp_emb_idx")
        elif "org_emb_idx" in inputs:
            emb_ids = inputs["org_emb_idx"]     # (B, # of different words)
            inputs.pop("org_emb_idx")
        else:
            emb_ids = None      # it means that this input is for the validation or test

        # print("\n\n")
        # print(f"Input keys: {inputs.keys()}")
        # print("\n\n")
        outputs = self.encoder(**inputs, output_hidden_states=True)  # outputs: (loss, logits, hidden_states)
        if 'labels' in inputs:
            logits = outputs[1]  # (B, N)
            last_hidden_states = outputs[2][-1]  # (B, N, D)
            labels = inputs['labels']
            masks = (labels != -100)
            hidden = []
            for i, last_batch_hidden in enumerate(last_hidden_states):
                if emb_ids is None:
                    pass
                else:
                    cur_emb_ids = emb_ids[i]        # (N, )
                    if torch.sum(cur_emb_ids) == 0:       # if there is no different word in the sentence
                        # hidden.append(torch.zeros(last_batch_hidden.shape[1], device=self.device))      # (D, )
                        continue
                    else:
                        not_zero = torch.where(cur_emb_ids != 0)[0]
                        hidden.append(torch.mean(last_batch_hidden[not_zero], 0))     # (B, D) --> (D, )
            if len(hidden) != 0:
                hidden = torch.stack(hidden)  # (B, D)

            return self.crf(logits, labels, masks), hidden
        else:
            last_hidden_states = outputs[1][-1]
            hidden = []
            for i, last_batch_hidden in enumerate(last_hidden_states):
                if emb_ids is None:
                    pass
                else:
                    cur_emb_ids = emb_ids[i]  # (N, )
                    if torch.sum(cur_emb_ids) == 0:  # if there is no different word in the sentence
                        # hidden.append(torch.zeros(last_batch_hidden.shape[1], device=self.device))      # (D, )
                        continue
                    else:
                        not_zero = torch.where(cur_emb_ids != 0)[0]
                        hidden.append(torch.mean(last_batch_hidden[not_zero], 0))  # (B, D) --> (D, )
            if len(hidden) != 0:
                hidden = torch.stack(hidden)  # (B, D)
            return None, hidden


class BERT_CRF_BothContrastive(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF_BothContrastive, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config,
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        if "inp_emb_idx" in inputs:
            emb_ids = inputs["inp_emb_idx"]     # (B, # of different words)
            inputs.pop("inp_emb_idx")
        elif "org_emb_idx" in inputs:
            emb_ids = inputs["org_emb_idx"]     # (B, # of different words)
            inputs.pop("org_emb_idx")
        else:
            emb_ids = None      # it means that this input is for the validation or test

        # print("\n\n")
        # print(f"Input keys: {inputs.keys()}")
        # print("\n\n")
        outputs = self.encoder(**inputs, output_hidden_states=True)  # outputs: (loss, logits, hidden_states)
        if 'labels' in inputs:
            logits = outputs[1]  # (B, N)
            last_hidden_states = outputs[2][-1]  # (B, N, D)
            labels = inputs['labels']
            masks = (labels != -100)

            # For the sentence contrastive learning
            hidden_s = torch.mean(last_hidden_states, 1)      # (B, D)        # look whole hidden states of all tokens in the sequences

            # For the word contrastive learning
            hidden_w = []
            for i, last_batch_hidden in enumerate(last_hidden_states):
                if emb_ids is None:
                    pass
                else:
                    cur_emb_ids = emb_ids[i]        # (N, )
                    if torch.sum(cur_emb_ids) == 0:       # if there is no different word in the sentence
                        # hidden_w.append(torch.zeros(last_batch_hidden.shape[1], device=self.device))      # (D, )
                        continue
                    else:
                        not_zero = torch.where(cur_emb_ids != 0)[0]
                        hidden_w.append(torch.mean(last_batch_hidden[not_zero], 0))     # (B, D) --> (D, )
            if len(hidden_w) != 0:
                hidden_w = torch.stack(hidden_w)  # (B, D)

            return self.crf(logits, labels, masks), hidden_s, hidden_w
        else:
            last_hidden_states = outputs[1][-1]
            hidden_s = torch.mean(last_hidden_states, 1)  # (B, D)        # look whole hidden states of all tokens in the sequences
            hidden_w = []
            for i, last_batch_hidden in enumerate(last_hidden_states):
                if emb_ids is None:
                    pass
                else:
                    cur_emb_ids = emb_ids[i]  # (N, )
                    if torch.sum(cur_emb_ids) == 0:  # if there is no different word in the sentence
                        # hidden_w.append(torch.zeros(last_batch_hidden.shape[1], device=self.device))      # (D, )
                        continue
                    else:
                        not_zero = torch.where(cur_emb_ids != 0)[0]
                        hidden_w.append(torch.mean(last_batch_hidden[not_zero], 0))  # (B, D) --> (D, )
            if len(hidden_w) != 0:
                hidden_w = torch.stack(hidden_w)  # (B, D)
            return None, hidden_s, hidden_w


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, model_name, device, config, cache_dir, hidden_dim):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(config.hidden_size, hidden_dim, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * hidden_dim, config.num_labels)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)
    
    def forward(self, **inputs):
        assert('labels' in inputs)
        last_hidden_state = self.encoder(output_hidden_states=True, **inputs)[2][-1]
        out = self.lstm(self.dropout1(last_hidden_state))[0]
        logits = self.fc(self.dropout2(out))
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(logits, labels, masks)












# My
class BERT_CRF_exBERT(nn.Module):
    def __init__(self, model_name, device, config, config_1, cache_dir):
        super(BERT_CRF_exBERT, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        
        self.extension_encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config_1, 
                                                                                 cache_dir=cache_dir, revision=model_revision,
                                                                                 use_auth_token=None)
        
        self.gate_ADD = nn.Linear(config.hidden_size, 1)
        
        # 이 부분은 불필요할 수 있습니다
        # self.output_layer = nn.Linear(config.hidden_size, config.num_labels)
        
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        assert('labels' in inputs)
        
        original_outputs = self.encoder(**inputs, output_hidden_states=True)
        original_hidden_states = original_outputs.hidden_states[-1]
        original_logits = original_outputs.logits
        
        extension_outputs = self.extension_encoder(**inputs, output_hidden_states=True)
        extension_hidden_states = extension_outputs.hidden_states[-1]
        extension_logits = extension_outputs.logits
        
        gate_values = torch.sigmoid(self.gate_ADD(original_hidden_states))
        
        combined_logits = gate_values * original_logits + (1 - gate_values) * extension_logits
        
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(combined_logits, labels, masks)


class BERT_CRF_원본(nn.Module):
    def __init__(self, model_name, device, config, cache_dir):
        super(BERT_CRF, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(model_name, from_tf=False, config=config, 
                                                                       cache_dir=cache_dir, revision=model_revision,
                                                                       use_auth_token=None)
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, **inputs):
        assert('labels' in inputs)
        logits = self.encoder(**inputs)[1]
        labels = inputs['labels']
        masks = (labels != -100)
        return self.crf(logits, labels, masks)

class BERT_CRF_fusion_emb(nn.Module):
    def __init__(self, model_name, device, config, config_ext, cache_dir):
        super(BERT_CRF_fusion_emb, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(
            model_name, from_tf=False, config=config, 
            cache_dir=cache_dir, revision='main', use_auth_token=None
        )
        
        # Extended embedding layer
        self.extended_embedding = nn.Embedding(config_ext.vocab_size, config.hidden_size)
        
        # Gating mechanism
        self.gate = nn.Linear(config.hidden_size * 2, 1)
        
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, extended_input_ids=None):
        # Original BERT output
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            output_hidden_states=True
        )
        sequence_output = outputs.hidden_states[-1]
        logits = outputs.logits

        # Extended embedding
        extended_embeddings = self.extended_embedding(extended_input_ids)

        # Gating mechanism
        gate_input = torch.cat([sequence_output, extended_embeddings], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))

        # Fuse embeddings
        fused_embeddings = gate_values * sequence_output + (1 - gate_values) * extended_embeddings

        # Use fused embeddings for final classification
        fused_logits = self.encoder.classifier(fused_embeddings)

        # CRF
        if labels is not None:
            masks = (labels != -100)
            return self.crf(fused_logits, labels, masks)
        else:
            masks = attention_mask.bool()
            return self.crf.decode(fused_logits, mask=masks)



class BERT_CRF_fusion_emb_frz(nn.Module):
    def __init__(self, model_name, device, config, config_ext, cache_dir, freeze_bert_crf=False):
        super(BERT_CRF_fusion_emb, self).__init__()
        self.device = device
        self.encoder = AutoModelForTokenClassification.from_pretrained(
            model_name, from_tf=False, config=config, 
            cache_dir=cache_dir, revision='main', use_auth_token=None
        )
        
        # Extended embedding layer (always trainable)
        self.extended_embedding = nn.Embedding(config_ext.vocab_size, config.hidden_size)
        
        # Gating mechanism (always trainable)
        self.gate = nn.Linear(config.hidden_size * 2, 1)
        
        # CRF layer
        self.crf = BIO_Tag_CRF(config.num_labels, device, batch_first=True)

        # Freeze BERT and CRF parameters if specified
        self.freeze_bert_crf = freeze_bert_crf
        if freeze_bert_crf:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.crf.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, extended_input_ids=None):
        # Original BERT output (frozen if freeze_bert_crf=True)
        with torch.no_grad() if self.freeze_bert_crf else torch.enable_grad():
            outputs = self.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                output_hidden_states=True
            )
        sequence_output = outputs.hidden_states[-1]
        logits = outputs.logits

        # Extended embedding (always computed with gradients)
        extended_embeddings = self.extended_embedding(extended_input_ids)

        # Gating mechanism (always computed with gradients)
        gate_input = torch.cat([sequence_output, extended_embeddings], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))

        # Fuse embeddings
        fused_embeddings = gate_values * sequence_output + (1 - gate_values) * extended_embeddings

        # Use fused embeddings for final classification
        fused_logits = self.encoder.classifier(fused_embeddings)

        # CRF (frozen if freeze_bert_crf=True)
        if labels is not None:
            masks = (labels != -100)
            with torch.no_grad() if self.freeze_bert_crf else torch.enable_grad():
                return self.crf(fused_logits, labels, masks)
        else:
            masks = attention_mask.bool()
            with torch.no_grad() if self.freeze_bert_crf else torch.enable_grad():
                return self.crf.decode(fused_logits, mask=masks)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]