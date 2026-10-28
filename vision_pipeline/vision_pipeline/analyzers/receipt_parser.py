# analyzers/receipt_parser.py — Donut-based version
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class ReceiptParserAnalyzer(BaseAnalyzer):
    name = "receipt_parser"

    MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

    def __init__(self):
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        self.processor = DonutProcessor.from_pretrained(self.MODEL_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def analyze(self, image, context):
        import torch
        import re
        from PIL import Image
        import numpy as np

        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_img = pil_img.convert("RGB")

        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = (
            sequence
            .replace(self.processor.tokenizer.eos_token, "")
            .replace(self.processor.tokenizer.pad_token, "")
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        parsed = self.processor.token2json(sequence)

        return AnalysisResult(self.name, {
            "entities": parsed,
            "raw_sequence_preview": sequence[:200],
            "model": self.MODEL_NAME,
        })
