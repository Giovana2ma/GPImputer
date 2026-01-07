#!/usr/bin/env python3
"""
Parse a GP run log (e.g. `resultados.log` or `teste4.log`) and extract the
best fitness per generation for each dataset / missing ratio / seed.

Outputs two files next to the input file: `<input>_parsed.csv` and
`<input>_parsed.json`.

Usage:
  python3 analysis/parse_results_log.py --input path/to/resultados.log

The CSV columns are: dataset,missing_ratio,seed,gen,best_fitness,f1
"""
import argparse
import json
import os
import re
from collections import defaultdict


GEN_RE = re.compile(r"^Gen\s+(\d+):\s*Best Fitness\s*=\s*([0-9]*\.?[0-9]+)(?:\s*\(F1\s*=\s*([0-9]*\.?[0-9]+)\))?", re.IGNORECASE)
DATASET_RE = re.compile(r"^DATASET:\s*(\S+)", re.IGNORECASE)
MISSING_RE = re.compile(r"^Missing ratio:\s*([0-9.]+%?)", re.IGNORECASE)
SEED_RE = re.compile(r"^Seed:\s*(\d+)", re.IGNORECASE)
SEED_ALT_RE = re.compile(r"SEED=(\d+)", re.IGNORECASE)


def parse_log(path):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # structure: results[dataset][missing_ratio][seed] = list of dicts {gen, best_fitness, f1}

    current_dataset = None
    current_missing = None
    current_seed = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = DATASET_RE.search(line)
            if m:
                current_dataset = m.group(1)
                # reset other context
                current_missing = None
                current_seed = None
                continue

            m = MISSING_RE.search(line)
            if m:
                current_missing = m.group(1)
                continue

            m = SEED_RE.search(line)
            if m:
                current_seed = m.group(1)
                continue

            # Some logs include 'Executando GP com SEED=42' style lines
            m = SEED_ALT_RE.search(line)
            if m and current_seed is None:
                current_seed = m.group(1)
                continue

            m = GEN_RE.search(line)
            if m and current_dataset is not None and current_missing is not None and current_seed is not None:
                gen = int(m.group(1))
                best_fitness = float(m.group(2))
                f1 = float(m.group(3)) if m.group(3) else None
                results[current_dataset][current_missing][current_seed].append({
                    "gen": gen,
                    "best_fitness": best_fitness,
                    "f1": f1,
                })

    return results


def results_to_csv_json(results, input_path):
    base = os.path.splitext(input_path)[0]
    csv_path = base + "_parsed.csv"
    json_path = base + "_parsed.json"

    # write CSV
    with open(csv_path, "w", encoding="utf-8") as out:
        out.write("dataset,missing_ratio,seed,gen,best_fitness,f1\n")
        for dataset in sorted(results.keys()):
            for missing in sorted(results[dataset].keys()):
                for seed in sorted(results[dataset][missing].keys(), key=lambda s: int(s)):
                    gens = sorted(results[dataset][missing][seed], key=lambda d: d["gen"])
                    for entry in gens:
                        f1 = '' if entry["f1"] is None else str(entry["f1"])
                        out.write(f"{dataset},{missing},{seed},{entry['gen']},{entry['best_fitness']},{f1}\n")

    # write JSON
    # convert defaultdict -> normal dict
    normal = {}
    for dataset, d_missing in results.items():
        normal[dataset] = {}
        for missing, d_seeds in d_missing.items():
            normal[dataset][missing] = {}
            for seed, entries in d_seeds.items():
                normal[dataset][missing][seed] = sorted(entries, key=lambda e: e["gen"])

    with open(json_path, "w", encoding="utf-8") as outj:
        json.dump(normal, outj, indent=2, ensure_ascii=False)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Parse GP run log and extract best fitness per generation")
    parser.add_argument("--input", "-i", required=True, help="Path to the log file (e.g. resultados.log)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        return

    results = parse_log(args.input)
    csv_path, json_path = results_to_csv_json(results, args.input)
    print(f"Parsed results written to: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
