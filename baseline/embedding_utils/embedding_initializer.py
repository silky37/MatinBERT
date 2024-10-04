import numpy as np
import torch
import torch.nn as nn
# from transformers import BertModel,BertForMaskedLM
# from model.classification_model import PretrainedTransformer
import logging

logger = logging.getLogger(__name__)


def transfer_embedding(transfer_model, d2p, type):
    # logger.info("Transfer representation from pretrained model : {}".format(type))

    def average():
        transfer_layer = transfer_model.bert.embeddings.word_embeddings if type == "average_input" else transfer_model.bert.cls.predictions.decoder
        embedding_layer = transfer_model.bert.embeddings.word_embeddings

        # print(embedding_layer.weight.shape)
        # print(transfer_layer.weight.shape)

        for key, values in d2p.items():
            embedding_id = values[0]
            transfer_ids = values[-1]
            try:
                transfer_embeddings = torch.cat([transfer_layer.weight[t_id].data.view(1, -1) for t_id in transfer_ids],
                                                dim=0)
                embedding_layer.weight.data[embedding_id] = torch.mean(transfer_embeddings, dim=0)

            except:
                logger.info("* random initialize on %s" % (embedding_id))
                pass

    def distill():
        return NotImplementedError


    if type == "average_input" or type == "average_output":
        average()

    elif type == "distill_input" or type == "distill_output":
        distill()





def transfer_fusion_embedding(model, d2p, init_type="subword", pretrained_embedding_path=None):
    logger.info(f"Initializing embeddings using {init_type} method.")

    original_embedding = model.encoder.bert.embeddings.word_embeddings
    extended_embedding = model.extended_embedding

    original_vocab_size = original_embedding.num_embeddings
    extended_vocab_size = extended_embedding.num_embeddings
    embedding_dim = original_embedding.embedding_dim

    if init_type not in ["random", "subword", "pretrained"]:
        raise ValueError("init_type must be 'random', 'subword', or 'pretrained'")

    if init_type == "pretrained" and pretrained_embedding_path is None:
        raise ValueError("pretrained_embedding_path must be provided when init_type is 'pretrained'")

    external_embeddings = None
    if init_type == "pretrained":
        try:
            external_embeddings = np.load(pretrained_embedding_path)
            logger.info(f"Loaded external embeddings from {pretrained_embedding_path}")
            if len(external_embeddings) != extended_vocab_size:
                raise ValueError(f"External embeddings size ({len(external_embeddings)}) does not match the extended vocabulary size ({extended_vocab_size})")
        except Exception as e:
            logger.error(f"Failed to load external embeddings: {str(e)}")
            raise

    # Update original_embedding for extended vocab
    for token, (new_id, subwords, subword_ids) in d2p.items():
        if new_id < original_vocab_size:
            continue
        subword_embeddings = [original_embedding.weight.data[i] for i in subword_ids if i < original_vocab_size]
        if subword_embeddings:
            avg_embedding = torch.stack(subword_embeddings).mean(dim=0)
            original_embedding.weight.data[new_id] = avg_embedding
        else:
            logger.warning(f"No valid subwords found for token {token} in original vocab. Using random initialization.")
            original_embedding.weight.data[new_id] = torch.randn(embedding_dim)

    # Initialize extended_embedding
    for token, (new_id, subwords, subword_ids) in d2p.items():
        if new_id < original_vocab_size:
            continue

        if init_type == "random":
            extended_embedding.weight.data[new_id - original_vocab_size] = torch.randn(embedding_dim)

        elif init_type == "subword":
            subword_embeddings = [original_embedding.weight.data[i] for i in subword_ids if i < original_vocab_size]
            if subword_embeddings:
                avg_embedding = torch.stack(subword_embeddings).mean(dim=0)
                extended_embedding.weight.data[new_id - original_vocab_size] = avg_embedding
            else:
                logger.warning(f"No valid subwords found for token {token}. Using random initialization.")
                extended_embedding.weight.data[new_id - original_vocab_size] = torch.randn(embedding_dim)

        elif init_type == "pretrained":
            if external_embeddings is not None and new_id - original_vocab_size < len(external_embeddings):
                extended_embedding.weight.data[new_id - original_vocab_size] = torch.tensor(external_embeddings[new_id - original_vocab_size])
            else:
                logger.warning(f"No external embedding found for token {token}. Using average of pretrained embeddings.")
                subword_embeddings = [original_embedding.weight.data[i] for i in subword_ids if i < original_vocab_size]
                if subword_embeddings:
                    avg_embedding = torch.stack(subword_embeddings).mean(dim=0)
                    extended_embedding.weight.data[new_id - original_vocab_size] = avg_embedding
                else:
                    logger.warning(f"No valid subwords found for token {token}. Using random initialization.")
                    extended_embedding.weight.data[new_id - original_vocab_size] = torch.randn(embedding_dim)

    logger.info(f"Embedding initialization completed. Total tokens in extended embedding: {extended_vocab_size}")

    return model