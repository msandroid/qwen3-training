#!/usr/bin/env python3
"""
GGUF Model Evaluation Script

Evaluates translation quality of GGUF models using standard metrics:
- BLEU (Bilingual Evaluation Understudy)
- chrF (Character n-gram F-score)
- TER (Translation Edit Rate)
- COMET (optional, neural metric)

Requirements:
- llama-cpp-python
- sacrebleu
- unbabel-comet (optional)

Usage:
    python evaluate_gguf.py --model model.gguf --test-data test.jsonl
    python evaluate_gguf.py --model model.gguf --test-data test.jsonl --max-samples 100
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


def check_dependencies() -> bool:
    """Check required packages; return True if all present."""
    missing = []
    
    try:
        import llama_cpp
    except ImportError:
        missing.append("llama-cpp-python")
    
    try:
        import sacrebleu
    except ImportError:
        missing.append("sacrebleu")
    
    if missing:
        print(f"Missing dependencies: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


# Language code to full name mapping
LANG_NAMES = {
    "en": "English", "ja": "Japanese", "sw": "Swahili", "ha": "Hausa",
    "yo": "Yoruba", "ig": "Igbo", "am": "Amharic", "zu": "Zulu",
    "xh": "Xhosa", "af": "Afrikaans", "so": "Somali", "rw": "Kinyarwanda",
    "sn": "Shona", "tw": "Twi", "ee": "Ewe", "wo": "Wolof",
    "ny": "Chichewa", "ti": "Tigrinya", "nso": "Northern Sotho",
    "tn": "Tswana", "om": "Oromo", "ve": "Venda", "nd": "Ndebele",
    "rn": "Kirundi", "fr": "French", "pt": "Portuguese", "es": "Spanish",
    "de": "German", "zh": "Chinese", "ko": "Korean", "ar": "Arabic",
    "ru": "Russian", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay",
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "tr": "Turkish", "uk": "Ukrainian", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "el": "Greek", "he": "Hebrew", "fa": "Persian",
}


def to_lang_name(code_or_name: str) -> str:
    """Return full language name for prompt."""
    if not code_or_name:
        return "English"
    key = code_or_name.strip().lower()
    return LANG_NAMES.get(key, code_or_name.strip())


def parse_language_pair(user_content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (source_lang, target_lang) from user prompt."""
    
    # "Translate from English to Swahili"
    m = re.search(r"Translate from\s+(\w+)\s+to\s+(\w+)", user_content, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    
    # "English to Swahili:"
    m = re.search(r"(\w+)\s+to\s+(\w+):", user_content, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    
    return None, None


def extract_source_and_reference(
    user_content: str,
    assistant_content: str
) -> tuple[Optional[str], Optional[str]]:
    """Extract source text and reference translation."""
    
    if "\n\n" not in user_content:
        return None, None
    
    source = user_content.split("\n\n", 1)[-1].strip()
    if not source:
        return None, None
    
    ref = (assistant_content or "").strip()
    if not ref:
        return None, None
    
    return source, ref


def load_test_set(
    test_data_path: Path,
    max_samples: int,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """
    Load test.jsonl and return (sources, references, lang_pairs).
    
    Args:
        test_data_path: Path to test JSONL file
        max_samples: Maximum number of samples to load
        source_lang: Filter by source language (optional)
        target_lang: Filter by target language (optional)
    
    Returns:
        Tuple of (source texts, reference translations, language pairs)
    """
    sources = []
    references = []
    lang_pairs: list[tuple[str, str]] = []
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(sources) >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            messages = data.get("messages", [])
            if len(messages) < 2:
                continue
            
            # Extract user and assistant content
            user_content = ""
            assistant_content = ""
            for m in messages:
                role = (m.get("role") or "").strip().lower()
                content = (m.get("content") or "").strip()
                if role == "user":
                    user_content = content
                elif role == "assistant":
                    assistant_content = content
            
            if not user_content or not assistant_content:
                continue
            
            src, ref = extract_source_and_reference(user_content, assistant_content)
            if src is None:
                continue
            
            # Parse language pair
            sl, tl = parse_language_pair(user_content)
            
            # Filter by language if specified
            if source_lang and target_lang:
                if sl is None or tl is None:
                    continue
                sln = to_lang_name(sl).lower()
                tln = to_lang_name(tl).lower()
                want_src = to_lang_name(source_lang).lower()
                want_tgt = to_lang_name(target_lang).lower()
                if sln != want_src or tln != want_tgt:
                    continue
            
            # Determine language pair
            if sl and tl:
                pair = (to_lang_name(sl), to_lang_name(tl))
            else:
                pair = ("English", "Unknown")
            
            sources.append(src)
            references.append(ref)
            lang_pairs.append(pair)
    
    return sources, references, lang_pairs


def strip_thinking_block(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3 model output.
    
    Handles:
    - Complete blocks: <think>reasoning</think>translation
    - Incomplete blocks: <think>reasoning... (no closing tag)
    """
    if not text:
        return ""
    
    # Complete block - extract text after it
    if "</think>" in text:
        result = text.split("</think>")[-1].strip()
        if result:
            return result
    
    # Incomplete block - no translation available
    if "<think>" in text and "</think>" not in text:
        return ""
    
    # No thinking block
    if "<think>" not in text:
        return text.strip()
    
    return text.strip()


def run_inference(
    model_path: Path,
    sources: list[str],
    lang_pairs: list[tuple[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    suppress_thinking: bool = True,
) -> list[str]:
    """
    Run inference using llama-cpp-python.
    
    Args:
        model_path: Path to GGUF model file
        sources: List of source texts to translate
        lang_pairs: List of (source_lang, target_lang) tuples
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        suppress_thinking: Add system prompt to suppress <think> blocks
    
    Returns:
        List of generated translations
    """
    from llama_cpp import Llama
    
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    
    hypotheses = []
    
    # System prompt to suppress thinking mode
    system_prompt = (
        "You are a direct translation assistant. "
        "Output ONLY the translation without any explanation, reasoning, or thinking. "
        "Do not use <think> tags. Respond with just the translated text."
    )
    
    print(f"Running inference on {len(sources)} samples...")
    
    for i, (src, (src_lang, tgt_lang)) in enumerate(zip(sources, lang_pairs)):
        prompt = f"Translate from {src_lang} to {tgt_lang}:\n\n{src}"
        text = ""
        
        # Build messages
        messages = []
        if suppress_thinking:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Try chat completion first
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>"],
            )
            choice = (out.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = (msg.get("content") or "").strip()
            text = strip_thinking_block(text)
        except Exception as e:
            print(f"  Chat completion error for sample {i}: {e}")
        
        # Fallback to raw completion
        if not text:
            if suppress_thinking:
                qwen_prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            else:
                qwen_prompt = (
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            
            try:
                out = llm(
                    qwen_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|im_end|>"],
                )
                raw = (out.get("choices") or [{}])[0].get("text") or ""
                text = strip_thinking_block(raw)
            except Exception as e:
                print(f"  Raw completion error for sample {i}: {e}")
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(sources)} samples...")
        
        hypotheses.append(text)
    
    return hypotheses


def compute_metrics(
    hypotheses: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
    compute_comet: bool = False,
) -> dict:
    """
    Compute translation quality metrics.
    
    Args:
        hypotheses: Generated translations
        references: Reference translations
        sources: Source texts (required for COMET)
        compute_comet: Whether to compute COMET score
    
    Returns:
        Dictionary with metric scores
    """
    import sacrebleu
    
    refs = [references]
    
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    ter = sacrebleu.corpus_ter(hypotheses, refs)
    
    scores = {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "ter": round(ter.score, 2),
    }
    
    # COMET (optional)
    if compute_comet and sources:
        try:
            from comet import download_model, load_from_checkpoint
            
            print("Computing COMET score...")
            model_path = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]
            output = model.predict(data, batch_size=8, gpus=0)
            scores["comet"] = round(output.system_score, 4)
        except Exception as e:
            print(f"COMET computation skipped: {e}")
    
    return scores


def print_results(
    scores: dict,
    hypotheses: list[str],
    sources: list[str],
    references: list[str],
    lang_pairs: list[tuple[str, str]],
    num_samples: int = 5,
):
    """Print evaluation results."""
    
    non_empty = sum(1 for h in hypotheses if h.strip())
    empty_count = len(hypotheses) - non_empty
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total samples: {len(hypotheses)}")
    print(f"  Non-empty outputs: {non_empty} ({100*non_empty/len(hypotheses):.1f}%)")
    print(f"  Empty outputs: {empty_count}")
    print(f"\n  BLEU : {scores.get('bleu', 0):.2f}")
    print(f"  chrF : {scores.get('chrf', 0):.2f}")
    print(f"  TER  : {scores.get('ter', 0):.2f}")
    if "comet" in scores:
        print(f"  COMET: {scores['comet']:.4f}")
    
    # Quality assessment
    bleu = scores.get("bleu", 0)
    if bleu >= 25:
        quality = "High Quality"
    elif bleu >= 20:
        quality = "Production Ready"
    elif bleu >= 15:
        quality = "Minimum Viable"
    else:
        quality = "Needs Improvement"
    print(f"\n  Quality Assessment: {quality}")
    
    # Sample outputs
    print(f"\n{'='*60}")
    print("Sample Outputs")
    print(f"{'='*60}")
    
    for i in range(min(num_samples, len(sources))):
        src_lang, tgt_lang = lang_pairs[i]
        print(f"\n[{i+1}] {src_lang} -> {tgt_lang}")
        print(f"  Source: {sources[i][:80]}{'...' if len(sources[i]) > 80 else ''}")
        print(f"  Output: {hypotheses[i][:80] if hypotheses[i] else '(empty)'}")
        print(f"  Reference: {references[i][:80]}{'...' if len(references[i]) > 80 else ''}")


def save_results(
    output_dir: Path,
    scores: dict,
    hypotheses: list[str],
    sources: list[str],
    references: list[str],
    lang_pairs: list[tuple[str, str]],
    model_path: str,
):
    """Save evaluation results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save source, reference, hypothesis files
    with open(output_dir / "source.txt", "w", encoding="utf-8") as f:
        for s in sources:
            f.write(s.replace("\n", " ") + "\n")
    
    with open(output_dir / "reference.txt", "w", encoding="utf-8") as f:
        for r in references:
            f.write(r.replace("\n", " ") + "\n")
    
    with open(output_dir / "hypothesis.txt", "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(h.replace("\n", " ") + "\n")
    
    # Save JSON results
    non_empty = sum(1 for h in hypotheses if h.strip())
    results = {
        "model": model_path,
        "num_samples": len(sources),
        "non_empty_outputs": non_empty,
        "empty_outputs": len(hypotheses) - non_empty,
        "scores": scores,
        "samples": [
            {
                "source": sources[i],
                "hypothesis": hypotheses[i],
                "reference": references[i],
                "lang_pair": f"{lang_pairs[i][0]} -> {lang_pairs[i][1]}"
            }
            for i in range(min(20, len(sources)))
        ],
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="GGUF Model Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_gguf.py --model model.gguf --test-data test.jsonl
  
  # Limit samples
  python evaluate_gguf.py --model model.gguf --test-data test.jsonl --max-samples 100
  
  # Filter by language pair
  python evaluate_gguf.py --model model.gguf --test-data test.jsonl \\
    --source-lang English --target-lang Swahili
  
  # Include COMET score
  python evaluate_gguf.py --model model.gguf --test-data test.jsonl --comet
        """
    )
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to GGUF model file")
    
    # Data
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum samples to evaluate")
    parser.add_argument("--source-lang", type=str,
                        help="Filter by source language")
    parser.add_argument("--target-lang", type=str,
                        help="Filter by target language")
    
    # Inference
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--n-ctx", type=int, default=2048,
                        help="Context window size")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers (-1 = all, 0 = CPU only)")
    parser.add_argument("--no-suppress-thinking", action="store_true",
                        help="Disable system prompt that suppresses <think> blocks")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--comet", action="store_true",
                        help="Compute COMET score (requires unbabel-comet)")
    
    # Execution
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip dependency checks")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Validate test data path
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}")
        sys.exit(1)
    
    # Load test set
    print("Loading test set...")
    sources, references, lang_pairs = load_test_set(
        test_data_path,
        args.max_samples,
        args.source_lang,
        args.target_lang,
    )
    
    if not sources:
        print("No samples found. Check test data format or language filters.")
        sys.exit(1)
    
    print(f"Loaded {len(sources)} samples")
    
    # Run inference
    hypotheses = run_inference(
        model_path,
        sources,
        lang_pairs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        suppress_thinking=not args.no_suppress_thinking,
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    scores = compute_metrics(
        hypotheses,
        references,
        sources=sources if args.comet else None,
        compute_comet=args.comet,
    )
    
    # Print results
    print_results(scores, hypotheses, sources, references, lang_pairs)
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(
        output_dir,
        scores,
        hypotheses,
        sources,
        references,
        lang_pairs,
        str(model_path),
    )


if __name__ == "__main__":
    main()
