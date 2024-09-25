from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .interfaces import Model
from .dataset import Item


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(
    transformer,
    tokenizer,
    beginning: str,
    temperature: float = 0,
    max_new_tokens: int = 64,
    do_sample=False,
    top_p: Optional[float] = None,
) -> str:
    inputs = tokenizer(beginning, return_tensors="pt").to(device)
    generated_ids = transformer.generate(
        **inputs,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def compute_token_log_likelihood(
    transformer, tokenizer, text: str, add_special_tokens: bool = True
) -> dict:
    """Simple but inefficient method to calculate the log likelihood of a given text."""
    # Tokenize the text
    encoded_text = tokenizer(
        [text],
        padding=False,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    token_indices = encoded_text.input_ids[0][:]

    # Get logits from the model at every position before EOS
    with torch.no_grad():
        logits = transformer(**encoded_text).logits[0]
    # Calculate log likelihoods from the logits
    log_ls = F.log_softmax(logits, dim=1)

    # Get log likelihoods of actual tokens
    token_log_ls = []
    tokens = []
    token_ids = []
    for token_pos in range(1, len(token_indices)):
        # Note: To calculate the likelihood of the given sequence,
        # at a given position we need to check which likelihood where predicted
        # in the last step
        token_id = token_indices[token_pos]
        token_log_ls.append(log_ls[token_pos - 1, token_id].item())
        token_ids.append(token_id)
        tokens.append(tokenizer.convert_ids_to_tokens([token_id])[0])

    return {"log_likelihoods": token_log_ls, "tokens": tokens, "token_ids": token_ids}


class Transformer(Model):
    """Class to use HuggingFace transformer models"""

    def __init__(
        self,
        model_name,
        lazy_loading: bool = True,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        self.lazy_loading = lazy_loading

        if name is not None:
            self.name = name
        else:
            self.name = model_name

        if not self.lazy_loading:
            self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_kwargs
        )
        model.to(device)
        if self.model_name.startswith("apple/OpenELM"):
            # tokenizer_name = 'NousResearch/Llama-2-7b-hf'
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
        else:
            tokenizer_name = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, **self.tokenizer_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()  # Set the model to evaluation mode

        return model, tokenizer

    def get_model_and_tokenizer(self):
        if self.lazy_loading:
            model, tokenizer = self._load_model()
        else:
            model, tokenizer = self.model, self.tokenizer
        return model, tokenizer

    def compute_continuations(
        self,
        items: List[Item],
        **kwargs,
    ) -> List[str]:
        """Generate continuations of the prompts of all items"""
        model, tokenizer = self.get_model_and_tokenizer()

        continuations = []

        print("Computing text continuations ...")
        for item in tqdm(items):
            text = generate(
                transformer=model,
                tokenizer=tokenizer,
                beginning=item.get_continuation_prompt(),
                **kwargs,
            )
            # Only consider the part after the given text
            text = text[len(item.prompt) :]

            continuations.append(text)
        return continuations

    def compute_option_log_likelihoods(
        self,
        items: List[Item],
        **kwargs,
    ) -> List[List[float]]:
        model, tokenizer = self.get_model_and_tokenizer()

        # Initialize return list
        log_likelihoods = []

        print("Computing option log likelihoods ...")
        for item in tqdm(items):
            input_texts = item.get_statements()

            # Tokenize the combined texts
            encoded_inputs = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            # Get logits from the model
            with torch.no_grad():
                logits = model(**encoded_inputs, **kwargs).logits

            # Calculate log probabilities from the logits for each token
            log_probs = F.log_softmax(logits, dim=-1)

            if tokenizer.padding_side == "right":
                # Need to offset by one since previous position contains prediction
                # for current token
                log_probs = log_probs[:, :-1]

                input_ids = encoded_inputs["input_ids"][:, 1:]
                attention_mask = encoded_inputs["attention_mask"][:, 1:]

                # Get log probabilities of actual tokens
                token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(
                    -1
                )

                # Set irrelevant entries at the end (from padding) to zero
                masked_log_probs = (
                    (token_log_probs * attention_mask).detach().cpu().numpy()
                )

            else:
                # Offset by one since likelihoods refer to next position
                log_probs = log_probs[:, :-1]
                input_ids = encoded_inputs["input_ids"][:, 1:]
                attention_mask = encoded_inputs["attention_mask"]

                # Note: We don't offset the attention mask to exclude
                # likelihoods of BOS tokens for padded sequences
                # (which can be quite low)
                attention_mask = attention_mask[:, :-1]

                token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(
                    -1
                )
                masked_log_probs = (
                    (token_log_probs * attention_mask).detach().cpu().numpy()
                )

                # Note: Setting offsets to 0 leads to computing log likelihoods
                # over the whole sequence including the prompt (useful for debugging)
                offsets = np.argmax(attention_mask.detach().cpu().numpy(), axis=1)
                masked_log_probs = [
                    lls[offset:] for lls, offset in zip(masked_log_probs, offsets)
                ]

                # TODO Apply padding to have same format as for right padding?
                # (Then the conversion to tokens can be done below)

            item_log_likelihoods = [np.sum(lls) for lls in masked_log_probs]
            log_likelihoods.append(item_log_likelihoods)
        return log_likelihoods


class SimpleTransformer(Transformer):
    """Class to use HuggingFace transformer models with a simple-to-verify
    inference method

    Note that this class is maximized for understandability but isn't efficient!"""

    def compute_option_log_likelihoods(
        self,
        items: List[Item],
        subtract_prompt: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        model, tokenizer = self.get_model_and_tokenizer()

        log_likelihoods = []

        print("Computing option log likelihoods ...")
        for item in tqdm(items):
            input_texts = item.get_statements()

            if subtract_prompt:
                # .rstrip() is to avoid that a whitespace token with low likelihood
                # is considered here (which could lead to overall positive values
                # in the total!)
                prompt_log_likelihood = np.sum(
                    compute_token_log_likelihood(
                        transformer=model,
                        tokenizer=tokenizer,
                        text=item.prompt.rstrip(),
                        add_special_tokens=True,
                    )["log_likelihoods"]
                )
            else:
                prompt_log_likelihood = 0  # dummy with no effect

            option_log_likelihoods = []
            for text in input_texts:
                # First compute the log likelihood of the whole sequence
                log_likelihood = np.sum(
                    compute_token_log_likelihood(
                        transformer=model,
                        tokenizer=tokenizer,
                        text=text,
                        add_special_tokens=True,
                    )["log_likelihoods"]
                )

                # Now remove the part from the prompt
                option_log_likelihood = log_likelihood - prompt_log_likelihood

                option_log_likelihoods.append(option_log_likelihood)

            log_likelihoods.append(option_log_likelihoods)

        return log_likelihoods
