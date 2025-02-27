import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM


class EvaluationModel(DeepEvalBaseLLM):
    def __init__(self, name: str, path: str) -> None:
        
        # Example name: "Llama-3 8B"
        # Example path: "meta-llama/Meta-Llama-3-8B-Instruct"
        self.name = name
        self.path = path
        
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            path# ,
            # device_map="auto",
            # quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path
        )

        self.model = model_4bit
        self.tokenizer = tokenizer
        self.device = "cuda"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_cache=True,
            device_map="auto",
            # max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.name