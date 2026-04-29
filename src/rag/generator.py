"""
LLM Generator module untuk RAG pipeline.

Mendukung 2 backend:
1. GGUF (llama-cpp-python) → RAGGenerator
   - Model: Qwen3-4B-Instruct-Q8_0.gguf, dll.
2. HuggingFace Transformers → HFRAGGenerator
   - Model: Qwen/Qwen3-4B-Thinking-2507-FP8, dll.
   - Otomatis mem-parse <think>...</think> dan mengembalikan jawaban akhir.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

try:
    from llama_cpp import Llama  # type: ignore[import-not-found]
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[misc]
    _LLAMA_CPP_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    _HF_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment, misc]
    AutoTokenizer = None  # type: ignore[assignment, misc]
    _HF_AVAILABLE = False

_THINK_CLOSE_TOKEN_ID = 151668

logger = logging.getLogger(__name__)

DEFAULT_GENERATOR_MODEL_PATH = "models/Qwen3-4B-Instruct-Q8_0.gguf"

SYSTEM_PROMPT = (
    "Anda adalah asisten AI yang bertugas menjawab pertanyaan berdasarkan "
    "konteks dokumen yang diberikan. Gunakan informasi dari konteks untuk "
    "memberikan jawaban yang akurat dan relevan. Jika konteks tidak memuat "
    "informasi yang cukup untuk menjawab pertanyaan, nyatakan bahwa Anda "
    "tidak memiliki informasi yang memadai. Jawab dalam Bahasa Indonesia "
    "dengan jelas dan ringkas."
)


class RAGGenerator:
    """
    Generator untuk menghasilkan jawaban RAG menggunakan GGUF LLM.

    Menerima query + list of context strings, membangun chat prompt,
    dan menghasilkan jawaban melalui create_chat_completion.
    """

    def __init__(
        self,
        model: Any,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """
        Args:
            model: Llama instance yang sudah di-load
            max_tokens: Maksimum token output
            temperature: Sampling temperature (0 = deterministic)
            top_p: Nucleus sampling threshold
            repeat_penalty: Penalti untuk token berulang
            system_prompt: System prompt untuk LLM
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.system_prompt = system_prompt

        logger.info("RAGGenerator initialized")
        logger.info(f"  - max_tokens    : {max_tokens}")
        logger.info(f"  - temperature   : {temperature}")
        logger.info(f"  - top_p         : {top_p}")
        logger.info(f"  - repeat_penalty: {repeat_penalty}")

    def build_messages(
        self,
        query: str,
        contexts: List[str],
    ) -> List[Dict[str, str]]:
        """
        Bangun list messages untuk chat completion.

        Args:
            query: Pertanyaan dari user
            contexts: List teks konteks yang sudah di-retrieve

        Returns:
            List of {"role": ..., "content": ...} dicts
        """
        context_block = "\n\n".join(
            f"[Konteks {i + 1}]\n{ctx.strip()}"
            for i, ctx in enumerate(contexts)
        )

        user_content = (
            f"Konteks:\n{context_block}\n\n"
            f"Pertanyaan: {query}\n\n"
            f"Jawaban:"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        query: str,
        contexts: List[str],
    ) -> str:
        """
        Generate jawaban dari query dan konteks.

        Args:
            query: Pertanyaan user
            contexts: List teks konteks dari retrieval

        Returns:
            Jawaban yang di-generate sebagai string
        """
        if not contexts:
            logger.warning("Tidak ada konteks yang diberikan ke generator")
            return "Tidak ada informasi konteks yang tersedia untuk menjawab pertanyaan ini."

        messages = self.build_messages(query, contexts)

        try:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stream=False,
            )
            answer: str = response["choices"][0]["message"]["content"]
            return answer.strip()

        except Exception as e:
            logger.error(f"Error saat generate: {e}")
            return ""


class HFRAGGenerator:
    """
    Generator untuk RAG menggunakan HuggingFace Transformers.

    Dioptimalkan untuk Qwen3-4B-Thinking-2507-FP8 (thinking model):
    - Output diawali dengan blok <think>...</think> berisi chain-of-thought.
    - Jawaban akhir berada SETELAH token </think> (token ID 151668).
    - Blok thinking otomatis di-strip; hanya jawaban final yang dikembalikan.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        system_prompt: str = SYSTEM_PROMPT,
        return_thinking: bool = False,
    ):
        """
        Args:
            model           : AutoModelForCausalLM instance
            tokenizer       : AutoTokenizer instance
            max_new_tokens  : Maksimum token output (termasuk thinking)
            temperature     : Sampling temperature (rekomendasi: 0.6)
            top_p           : Nucleus sampling (rekomendasi: 0.95)
            top_k           : Top-K sampling (rekomendasi: 20)
            system_prompt   : System prompt untuk LLM
            return_thinking : Jika True, generate() return tuple (answer, thinking)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.return_thinking = return_thinking

        logger.info("HFRAGGenerator initialized (Qwen3-Thinking)")
        logger.info(f"  - max_new_tokens : {max_new_tokens}")
        logger.info(f"  - temperature    : {temperature}")
        logger.info(f"  - top_p          : {top_p}")
        logger.info(f"  - top_k          : {top_k}")
        logger.info(f"  - return_thinking: {return_thinking}")

    def build_messages(
        self,
        query: str,
        contexts: List[str],
    ) -> List[Dict[str, str]]:
        """Bangun messages list untuk chat template."""
        context_block = "\n\n".join(
            f"[Konteks {i + 1}]\n{ctx.strip()}"
            for i, ctx in enumerate(contexts)
        )
        user_content = (
            f"Konteks:\n{context_block}\n\n"
            f"Pertanyaan: {query}\n\n"
            f"Jawaban:"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        query: str,
        contexts: List[str],
    ) -> Union[str, tuple]:
        """
        Generate jawaban dari query dan konteks.

        Returns:
            str              : Jawaban final (jika return_thinking=False)
            (str, str) tuple : (answer, thinking) jika return_thinking=True
        """
        if not contexts:
            logger.warning("Tidak ada konteks yang diberikan ke HFRAGGenerator")
            empty = "Tidak ada informasi konteks yang tersedia untuk menjawab pertanyaan ini."
            return (empty, "") if self.return_thinking else empty

        messages = self.build_messages(query, contexts)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
            )
        except Exception as e:
            logger.error(f"Error saat generate: {e}")
            return ("", "") if self.return_thinking else ""

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse thinking content: token 151668 = </think>
        try:
            index = len(output_ids) - output_ids[::-1].index(_THINK_CLOSE_TOKEN_ID)
        except ValueError:
            index = 0

        thinking = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip()
        answer = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip()

        if answer:
            logger.info(f"Thinking length: {len(thinking)} chars | Answer length: {len(answer)} chars")
        else:
            logger.warning("Answer kosong setelah </think> — mengembalikan full output sebagai answer")
            answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            thinking = ""

        return (answer, thinking) if self.return_thinking else answer


def initialize_hf_generator(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507-FP8",
    max_new_tokens: int = 32768,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    system_prompt: str = SYSTEM_PROMPT,
    return_thinking: bool = False,
) -> Optional["HFRAGGenerator"]:
    """
    Load HuggingFace model dan return HFRAGGenerator.

    Args:
        model_name      : HuggingFace model name atau path lokal
        max_new_tokens  : Maksimum token output (termasuk thinking)
        temperature     : Sampling temperature (rekomendasi: 0.6)
        top_p           : Nucleus sampling (rekomendasi: 0.95)
        top_k           : Top-K sampling (rekomendasi: 20)
        system_prompt   : System prompt override
        return_thinking : Kembalikan juga thinking content

    Returns:
        HFRAGGenerator instance atau None jika gagal
    """
    if not _HF_AVAILABLE:
        logger.error(
            "transformers / torch tidak tersedia. "
            "Install: pip install transformers torch accelerate"
        )
        return None

    try:
        logger.info(f"Loading HF tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading HF model: {model_name} (torch_dtype=auto, device_map=auto)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        logger.info(f"✓ HF model loaded: {model_name}")
        if hasattr(model, 'hf_device_map'):
            logger.info(f"  - Device map: {model.hf_device_map}")

        return HFRAGGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_prompt=system_prompt,
            return_thinking=return_thinking,
        )

    except Exception as e:
        logger.error(f"Error loading HF model: {e}")
        return None


def initialize_gguf_generator(
    model_path: str = DEFAULT_GENERATOR_MODEL_PATH,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    n_batch: int = 512,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    system_prompt: str = SYSTEM_PROMPT,
    verbose: bool = False,
) -> Optional[RAGGenerator]:
    """
    Load GGUF chat model dan return RAGGenerator.

    Args:
        model_path: Path ke file .gguf (chat/instruct model)
        n_gpu_layers: Jumlah layer di GPU (-1 = semua, 0 = CPU only)
        n_ctx: Context length (tokens)
        n_batch: Batch size untuk prompt processing
        max_tokens: Maksimum token output
        temperature: Sampling temperature
        top_p: Nucleus sampling
        repeat_penalty: Penalti repetisi
        system_prompt: System prompt override
        verbose: Aktifkan verbose logging dari llama.cpp

    Returns:
        RAGGenerator instance atau None jika gagal
    """
    if not _LLAMA_CPP_AVAILABLE:
        logger.error(
            "llama-cpp-python tidak tersedia. "
            "Install: pip install llama-cpp-python"
        )
        return None

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error(f"Generator model tidak ditemukan: {model_path}")
        logger.error(
            "Download model GGUF chat/instruct dari HuggingFace dan "
            f"letakkan di: {model_path}"
        )
        return None

    try:
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"Loading generator model: {model_file.name} ({file_size_mb:.0f} MB)")
        logger.info(f"  - GPU Layers  : {n_gpu_layers} (-1 = semua)")
        logger.info(f"  - Context     : {n_ctx} tokens")
        logger.info(f"  - Batch       : {n_batch}")

        model = Llama(
            model_path=str(model_file),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            chat_format="chatml",
        )

        logger.info(f"✓ Generator model loaded: {model_file.name}")

        return RAGGenerator(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            system_prompt=system_prompt,
        )

    except Exception as e:
        logger.error(f"Error loading generator model: {e}")
        return None
