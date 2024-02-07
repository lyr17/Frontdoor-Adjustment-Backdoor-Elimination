import torch
import pandas as pd
import torch.nn as nn
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, \
    LlamaForCausalLM, AutoModelForCausalLM, BertTokenizer
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "bert-base-uncased",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        self.max_len = max_len
        # you can change huggingface model_config here
        if self.model_name == "bert":
            self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
            self.tokenizer = BertTokenizer.from_pretrained(path)
            self.to(self.device)
        elif self.model_name == "t5":
            self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config, device_map="balanced_low_0")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        elif self.model_name == "gpt-j":
            self.plm = AutoModelForCausalLM.from_pretrained(path, config=self.model_config, device_map="balanced_low_0", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(path, padding=True, truncation=True)
            self.tokenizer.pad_token = self.tokenizer.bos_token
        elif self.model_name == "llama2":
            self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config, device_map="auto", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(path, padding=True, truncation=True)
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.plm.config.pad_token_id = self.plm.config.bos_token_id
        
    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state  # batch_size, max_len, 768(1024)
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        text = [x for x in text if pd.isnull(x) == False]
        labels = labels[0:len(text)]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        if self.model_name == "t5":
            return layer.shared.weight
        elif self.model_name == "gpt-j":
            return layer.wte.weight
        elif self.model_name == "llama2":
            return layer.embed_tokens.weight
        else:
            return layer.embeddings.word_embeddings.weight
    
