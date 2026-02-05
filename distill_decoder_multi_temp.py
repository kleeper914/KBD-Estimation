# -*- coding: utf-8 -*-
"""
Decoder-only distillation with multiple temperatures.
使用基础模型参数，仅蒸馏解码器，生成不同温度下的收发知识库差异模型并保存。
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import SNR_to_noise, initNetParams, loss_function
import distillation as kd


def parse_temperature_list(value: str):
    temps = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        temps.append(float(part))
    if len(temps) == 0:
        raise ValueError("Temperature list is empty.")
    return temps


def freeze_base_modules(model: DeepSC):
    for name, param in model.named_parameters():
        if name.startswith("encoder") or name.startswith("channel_encoder") or name.startswith("channel_decoder"):
            param.requires_grad = False


def build_models(args, vocab_size):
    def _load_checkpoint_with_pe_resize(model, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=kd.device)
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

    # Teacher
    teacher_model = DeepSC(
        args.num_layers, vocab_size, vocab_size,
        vocab_size, vocab_size, args.d_model,
        args.num_heads, args.dff, 0.1
    )
    if os.path.exists(args.teacher_checkpoint):
        _load_checkpoint_with_pe_resize(teacher_model, args.teacher_checkpoint)
    else:
        teacher_model = initNetParams(teacher_model)
    teacher_model = teacher_model.to(kd.device)
    teacher_model.eval()

    # Student
    student_model = DeepSC(
        args.student_layers, vocab_size, vocab_size,
        vocab_size, vocab_size, args.d_model,
        args.num_heads, args.dff, 0.1
    )

    loaded_student = False
    if args.use_base_student and os.path.exists(args.teacher_checkpoint) and args.student_layers == args.num_layers:
        _load_checkpoint_with_pe_resize(student_model, args.teacher_checkpoint)
        loaded_student = True
    elif args.use_base_student and args.student_layers != args.num_layers:
        print(
            "[Warning] use_base_student is enabled but student_layers != num_layers; "
            "skip loading teacher checkpoint for student."
        )
    elif args.student_checkpoint and os.path.exists(args.student_checkpoint):
        _load_checkpoint_with_pe_resize(student_model, args.student_checkpoint)
        loaded_student = True

    if not loaded_student:
        student_model = initNetParams(student_model)

    student_model = student_model.to(kd.device)

    return teacher_model, student_model


def run_single_temperature(args):
    os.makedirs(args.output_path, exist_ok=True)
    kd.setup_seed(args.seed)

    # vocab
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["token_to_idx"])
    pad_idx = vocab["token_to_idx"]["<PAD>"]

    teacher_model, student_model = build_models(args, vocab_size)

    if args.freeze_base and args.distill_part == "decoder":
        freeze_base_modules(student_model)

    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    student_optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    train_eur = EurDataset("distill")
    train_iterator = torch.utils.data.DataLoader(
        train_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data,
    )

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        total_task_loss = 0.0
        total_distill_loss = 0.0
        batch_count = 0

        noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]

        pbar = tqdm(train_iterator, desc=f"T={args.temperature} Epoch {epoch+1}/{args.epochs}")
        for sents in pbar:
            sents = sents.to(kd.device)
            loss, task_loss, distill_loss = kd.distillation_step(
                student_model,
                teacher_model,
                sents,
                sents,
                noise_std,
                pad_idx,
                student_optimizer,
                criterion,
                args.channel,
                args.temperature,
                args.distill_weight,
                args.distill_part,
            )
            total_loss += loss
            total_task_loss += task_loss
            total_distill_loss += distill_loss
            batch_count += 1

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "task": f"{task_loss:.4f}",
                "distill": f"{distill_loss:.4f}",
            })

        avg_loss = total_loss / batch_count
        avg_task_loss = total_task_loss / batch_count
        avg_distill_loss = total_distill_loss / batch_count
        epoch_time = time.time() - epoch_start_time

        val_student_loss, val_teacher_loss, val_distill_loss = kd.validate_distillation(
            student_model, teacher_model, args, pad_idx, criterion, args.distill_part
        )

        if val_student_loss < best_loss:
            best_loss = val_student_loss
            best_epoch = epoch
            best_model_path = os.path.join(args.output_path, "best_student_model.pt")
            torch.save(student_model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_path, f"student_model_epoch{epoch+1}.pt")
            torch.save(student_model.state_dict(), checkpoint_path)

        print(
            f"[T={args.temperature}] Epoch {epoch+1}/{args.epochs} | "
            f"Time {epoch_time:.2f}s | "
            f"Train L {avg_loss:.5f} (Task {avg_task_loss:.5f}, Distill {avg_distill_loss:.5f}) | "
            f"Val L {val_student_loss:.5f} (Teacher {val_teacher_loss:.5f}, Distill {val_distill_loss:.5f})"
        )

    final_model_path = os.path.join(args.output_path, "final_student_model.pt")
    torch.save(student_model.state_dict(), final_model_path)

    hybrid_model = None
    if args.create_hybrid:
        os.makedirs(os.path.dirname(args.hybrid_output), exist_ok=True)
        hybrid_model = kd.create_hybrid_model(
            teacher_model,
            student_model,
            args.distill_part,
            vocab_size,
            args.d_model,
            args.dff,
            args.num_heads,
            args.num_layers,
            args.MAX_LENGTH,
            kd.device,
            args.hybrid_output,
        )

    if torch.is_tensor(best_loss):
        best_loss = float(best_loss.item())

    config = {
        "temperature": args.temperature,
        "distill_weight": args.distill_weight,
        "distill_part": args.distill_part,
        "d_model": args.d_model,
        "dff": args.dff,
        "num_heads": args.num_heads,
        "student_layers": args.student_layers,
        "teacher_layers": args.num_layers,
        "channel": args.channel,
        "best_loss": best_loss,
        "best_epoch": best_epoch + 1,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
        "create_hybrid": args.create_hybrid,
        "hybrid_output": args.hybrid_output if args.create_hybrid else None,
        "use_base_student": args.use_base_student,
        "freeze_base": args.freeze_base,
    }

    config_path = os.path.join(args.output_path, "distillation_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    return {
        "temperature": args.temperature,
        "output_path": args.output_path,
        "final_model": final_model_path,
        "hybrid_model": args.hybrid_output if args.create_hybrid else None,
        "config": config_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-file", default="europarl/vocab.json", type=str)
    parser.add_argument("--teacher-checkpoint", default="checkpoints/deepsc-Ideal/model.pt", type=str)
    parser.add_argument("--student-checkpoint", default="", type=str)
    parser.add_argument("--output-root", default="checkpoints/decoder_temp_multi", type=str)
    parser.add_argument("--channel", default="Ideal", type=str)
    parser.add_argument("--MAX-LENGTH", default=30, type=int)
    parser.add_argument("--MIN-LENGTH", default=4, type=int)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--temperatures", default="1.0,2.0,4.0,6.0,8.0", type=str)
    parser.add_argument("--distill-weight", default=0.7, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--student-layers", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--create-hybrid", action="store_true", default=True)
    parser.add_argument("--no-create-hybrid", action="store_false", dest="create_hybrid")
    parser.add_argument("--use-base-student", action="store_true", default=True)
    parser.add_argument("--no-use-base-student", action="store_false", dest="use_base_student")
    parser.add_argument("--freeze-base", action="store_true", default=True)
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")

    args = parser.parse_args()

    temps = parse_temperature_list(args.temperatures)

    if args.student_layers is None:
        args.student_layers = args.num_layers

    results = []
    for temp in temps:
        temp_tag = f"T{temp}"
        output_path = os.path.join(args.output_root, temp_tag)
        hybrid_output = os.path.join(output_path, "hybrid_decoder_distilled.pt")

        run_args = argparse.Namespace(
            vocab_file='/mnt/workspace/KBD-Estimation/data/' + args.vocab_file,
            teacher_checkpoint=args.teacher_checkpoint,
            student_checkpoint=args.student_checkpoint,
            output_path=output_path,
            channel=args.channel,
            MAX_LENGTH=args.MAX_LENGTH,
            MIN_LENGTH=args.MIN_LENGTH,
            d_model=args.d_model,
            dff=args.dff,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            epochs=args.epochs,
            temperature=temp,
            distill_weight=args.distill_weight,
            lr=args.lr,
            student_layers=args.student_layers,
            distill_part="decoder",
            create_hybrid=args.create_hybrid,
            hybrid_output=hybrid_output,
            seed=args.seed,
            use_base_student=args.use_base_student,
            freeze_base=args.freeze_base,
        )

        print("=" * 70)
        print(f"Decoder distillation with temperature {temp}")
        print("=" * 70)
        results.append(run_single_temperature(run_args))

    summary_path = os.path.join(args.output_root, "multi_temp_summary.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("All temperatures complete.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
