# -*- coding: utf-8 -*-
"""
Evaluate distilled models at multiple temperatures on the test set.
使用test数据集评估不同温度蒸馏模型的性能，反映温度系数带来的知识差异。
"""
import os
import json
import argparse
import pickle
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.transceiver import DeepSC
from utils import BleuScore, SeqtoText, SNR_to_noise, greedy_decode, val_step
from dataset import collate_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_temperature_list(value: str) -> List[float]:
    temps = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        temps.append(float(part))
    if len(temps) == 0:
        raise ValueError("Temperature list is empty.")
    return temps


class EurDatasetFromPath(Dataset):
    def __init__(self, data_root: str, split: str = "test"):
        data_dir = os.path.join(data_root, "europarl")
        data_path = os.path.join(data_dir, f"{split}_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_vocab(vocab_path: str) -> Dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_with_pe_resize(model: DeepSC, checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()

    for key in ["encoder.pos_encoding.pe", "decoder.pos_encoding.pe"]:
        if key in state and key in model_state:
            ckpt_pe = state[key]
            cur_pe = model_state[key]
            if ckpt_pe.shape != cur_pe.shape:
                if ckpt_pe.size(1) >= cur_pe.size(1):
                    state[key] = ckpt_pe[:, :cur_pe.size(1), :]
                else:
                    padded = cur_pe.clone()
                    padded[:, :ckpt_pe.size(1), :] = ckpt_pe
                    state[key] = padded

    model.load_state_dict(state, strict=False)


def resolve_model_config(args, config_path: str) -> Dict:
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    return {
        "d_model": config.get("d_model", args.d_model),
        "dff": config.get("dff", args.dff),
        "num_heads": config.get("num_heads", args.num_heads),
        "student_layers": config.get("student_layers", args.student_layers),
        "num_layers": config.get("teacher_layers", args.num_layers),
        "channel": config.get("channel", args.channel),
    }


def evaluate_bleu(
    model: DeepSC,
    test_iterator: DataLoader,
    sto_text: SeqtoText,
    snr_list: List[float],
    pad_idx: int,
    start_idx: int,
    channel: str,
    max_length: int,
) -> Tuple[List[float], float]:
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    score = []
    model.eval()

    with torch.no_grad():
        for snr in snr_list:
            word = []
            target_word = []
            noise_std = SNR_to_noise(snr)

            for sents in tqdm(test_iterator, desc=f"SNR={snr}", leave=False):
                sents = sents.to(device)
                target = sents

                out = greedy_decode(model, sents, noise_std, max_length, pad_idx, start_idx, channel)
                sentences = out.cpu().numpy().tolist()
                result_string = list(map(sto_text.sequence_to_text, sentences))
                word.extend(result_string)

                target_sent = target.cpu().numpy().tolist()
                result_string = list(map(sto_text.sequence_to_text, target_sent))
                target_word.extend(result_string)

            bleu_score = bleu_score_1gram.compute_blue_score(word, target_word)
            bleu_score = float(np.mean(np.array(bleu_score)))
            score.append(bleu_score)

    overall_bleu = float(np.mean(np.array(score))) if len(score) > 0 else 0.0
    return score, overall_bleu


def evaluate_loss(
    model: DeepSC,
    test_iterator: DataLoader,
    pad_idx: int,
    channel: str,
    eval_snr: float,
) -> float:
    criterion = nn.CrossEntropyLoss(reduction="none")
    noise_std = SNR_to_noise(eval_snr)

    total = 0.0
    model.eval()
    with torch.no_grad():
        for sents in test_iterator:
            sents = sents.to(device)
            loss = val_step(model, sents, sents, noise_std, pad_idx, criterion, channel)
            total += loss

    return total / len(test_iterator)


def evaluate_single_temperature(
    args,
    vocab: Dict,
    test_iterator: DataLoader,
    temp: float,
) -> Dict:
    temp_tag = f"T{temp}"
    temp_dir = os.path.join(args.output_root, temp_tag)
    config_path = os.path.join(temp_dir, "distillation_config.json")
    model_config = resolve_model_config(args, config_path)

    if args.use_hybrid:
        model_path = os.path.join(temp_dir, args.hybrid_name)
    else:
        model_path = os.path.join(temp_dir, args.checkpoint_name)

    if not os.path.exists(model_path):
        return {
            "temperature": temp,
            "status": "missing",
            "model_path": model_path,
        }

    vocab_size = len(vocab["token_to_idx"])

    model = DeepSC(
        model_config["student_layers"],
        vocab_size,
        vocab_size,
        args.max_length,
        args.max_length,
        model_config["d_model"],
        model_config["num_heads"],
        model_config["dff"],
        0.1,
    ).to(device)

    load_checkpoint_with_pe_resize(model, model_path)

    pad_idx = vocab["token_to_idx"]["<PAD>"]
    start_idx = vocab["token_to_idx"]["<START>"]
    end_idx = vocab["token_to_idx"]["<END>"]
    sto_text = SeqtoText(vocab["token_to_idx"], end_idx)

    bleu_per_snr, bleu_mean = evaluate_bleu(
        model,
        test_iterator,
        sto_text,
        args.snr_list,
        pad_idx,
        start_idx,
        model_config["channel"],
        args.max_length,
    )

    loss = evaluate_loss(
        model,
        test_iterator,
        pad_idx,
        model_config["channel"],
        args.eval_snr,
    )

    return {
        "temperature": temp,
        "status": "ok",
        "model_path": model_path,
        "channel": model_config["channel"],
        "bleu_per_snr": bleu_per_snr,
        "bleu_mean": bleu_mean,
        "eval_snr": args.eval_snr,
        "loss": loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="checkpoints/decoder_temp_multi", type=str)
    parser.add_argument("--temperatures", default="1.0,2.0,4.0,6.0,8.0", type=str)
    parser.add_argument("--checkpoint-name", default="best_student_model.pt", type=str)
    parser.add_argument("--use-hybrid", action="store_true", default=False)
    parser.add_argument("--hybrid-name", default="hybrid_decoder_distilled.pt", type=str)

    parser.add_argument("--data-root", default="/import/antennas/Datasets/hx301/", type=str)
    parser.add_argument("--vocab-file", default="europarl/vocab.json", type=str)
    parser.add_argument("--channel", default="Ideal", type=str)
    parser.add_argument("--max-length", default=30, type=int)
    parser.add_argument("--min-length", default=4, type=int)

    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--student-layers", default=4, type=int)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--snr-list", default="0,3,6,9,12,15,18", type=str)
    parser.add_argument("--eval-snr", default=9.0, type=float)

    parser.add_argument("--save-json", default=True, action="store_true")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false")
    parser.add_argument("--save-csv", default=True, action="store_true")
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false")

    args = parser.parse_args()

    temps = parse_temperature_list(args.temperatures)
    args.snr_list = [float(x.strip()) for x in args.snr_list.split(',') if x.strip()]

    vocab_path = os.path.join(args.data_root, args.vocab_file)
    vocab = load_vocab(vocab_path)

    test_dataset = EurDatasetFromPath(args.data_root, "test")
    test_iterator = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data,
    )

    results = []
    for temp in temps:
        print("=" * 70)
        print(f"Evaluating temperature {temp}")
        print("=" * 70)
        result = evaluate_single_temperature(args, vocab, test_iterator, temp)
        results.append(result)

        if result["status"] == "ok":
            print(
                f"T={temp} | BLEU_mean={result['bleu_mean']:.4f} | "
                f"Loss@SNR{result['eval_snr']}={result['loss']:.5f}"
            )
        else:
            print(f"T={temp} | Missing model: {result['model_path']}")

    os.makedirs(args.output_root, exist_ok=True)

    if args.save_json:
        summary_path = os.path.join(args.output_root, "temp_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Summary saved to: {summary_path}")

    if args.save_csv:
        csv_path = os.path.join(args.output_root, "temp_eval_summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("temperature,status,bleu_mean,loss,eval_snr,model_path\n")
            for r in results:
                if r["status"] != "ok":
                    f.write(f"{r['temperature']},missing,,,,{r['model_path']}\n")
                else:
                    f.write(
                        f"{r['temperature']},ok,{r['bleu_mean']:.6f},"
                        f"{r['loss']:.6f},{r['eval_snr']},{r['model_path']}\n"
                    )
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
