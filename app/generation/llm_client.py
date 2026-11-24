
import torch


class LLMClient:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, query, context):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strictly evidence-grounded ML assistant. "
                    "Use ONLY explicit facts in the provided context. "
                    "Never use prior knowledge, assumptions, or speculation. "
                    "If the answer is not explicitly supported by the context, "
                    "respond only with: "
                    "'The provided sources do not contain enough information "
                    "to answer this question.' "
                    "Do not add anything after this statement. "
                    "Cite source filenames for factual claims."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
