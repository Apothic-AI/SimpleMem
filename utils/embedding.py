"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
"""
from typing import List, Optional, Dict, Any
import numpy as np
import config
import os


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.embedding_base_url = getattr(config, "EMBEDDING_BASE_URL", None)
        self.embedding_api_key = (
            getattr(config, "EMBEDDING_API_KEY", None)
            or getattr(config, "OPENAI_API_KEY", None)
            or "dummy-key"
        )
        
        print(f"Loading embedding model: {self.model_name}")

        if self.embedding_base_url:
            self._init_remote_openai_embedding()
            return
        
        # Check if it's a Qwen3 model (through SentenceTransformers)
        if self.model_name.startswith("qwen3"):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    def _init_remote_openai_embedding(self):
        """Initialize remote embedding endpoint."""
        import httpx
        base = str(self.embedding_base_url).rstrip("/")
        urls = [f"{base}/embeddings", f"{base}/embedding"]
        if base.endswith("/v1"):
            base_no_v1 = base[:-3].rstrip("/")
            if base_no_v1:
                urls.extend([f"{base_no_v1}/embeddings", f"{base_no_v1}/embedding"])
        else:
            urls.append(f"{base}/v1/embeddings")
        deduped = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        self.model = httpx.Client(timeout=30.0)
        self.remote_urls = deduped
        self.dimension = int(getattr(config, "EMBEDDING_DIMENSION", 1024))
        self.model_type = "openai_compatible_remote"
        self.supports_query_prompt = False
        print(f"Using remote embedding endpoint: {self.embedding_base_url}")

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")
            
            # Initialize with optimization settings
            if self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        if self.model_type == "openai_compatible_remote":
            return self._encode_remote(texts)
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def _encode_remote(self, texts: List[str]) -> np.ndarray:
        """Encode texts using remote embeddings endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.embedding_api_key}"

        def _extract(payload: Any) -> List[float]:
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, list) and data:
                    emb = data[0].get("embedding")
                    if isinstance(emb, list):
                        if emb and isinstance(emb[0], list):
                            emb = emb[0]
                        return [float(x) for x in emb]
                emb = payload.get("embedding")
                if isinstance(emb, list):
                    if emb and isinstance(emb[0], list):
                        emb = emb[0]
                    return [float(x) for x in emb]
            if isinstance(payload, list) and payload:
                first = payload[0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    emb = first["embedding"]
                    if emb and isinstance(emb[0], list):
                        emb = emb[0]
                    return [float(x) for x in emb]
            raise ValueError("Unexpected embedding response shape")

        vectors = []
        for text in texts:
            last_exc = None
            for url in self.remote_urls:
                try:
                    response = self.model.post(
                        url,
                        headers=headers,
                        json={"model": self.model_name, "input": text},
                    )
                    response.raise_for_status()
                    vectors.append(_extract(response.json()))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.size == 0:
            raise ValueError("Embedding API returned empty vectors")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        array = array / norms
        self.dimension = int(array.shape[1])
        return array
