import argparse
import concurrent.futures
import glob
import json
import logging
import os

import tiktoken
import tqdm


def process_file(input_file, output_folder):
    """
    Process a single file:
      - Use GPT2 tokenizer to detokenize each line's tokens;
      - Create a new JSON object (preserve cluster_id, add text);
      - Write to a new .detokenized.parquet file;
      - Return the filename and total token count for that file.
    """
    output_file = os.path.join(
        output_folder,
        os.path.basename(input_file).replace(".tokenized.jsonl", ".detokenized.jsonl"),
    )
    os.makedirs(output_folder, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens_file = 0

    try:
        with (
            open(input_file, "r", encoding="utf-8") as fin,
            open(output_file, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error in file {input_file}: {e}")
                    continue

                tokens = data.get("tokens", [])
                token_count = data.get("token_count", len(tokens))
                total_tokens_file += token_count

                # Detokenize tokens
                text = tokenizer.decode(tokens)

                # Generate new JSON object
                new_data = {}
                if "cluster_id" in data:
                    new_data["cluster_id"] = data["cluster_id"]
                new_data["text"] = text
                new_data["token_count"] = token_count

                fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")

    return input_file, total_tokens_file


def process_folder_parallel(input_folder, output_folder, num_workers):
    """
    Find all .tokenized.jsonl files in the specified folder and process them in parallel:
      - Start a process for each file;
      - Display overall file processing progress using tqdm;
      - Accumulate the token count from all files.
    """
    tokenized_files = glob.glob(os.path.join(input_folder, "*.tokenized.jsonl"))
    if not tokenized_files:
        logging.warning("No .tokenized.jsonl files found in the specified folder.")
        return

    total_tokens_all = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit processing tasks for all files
        futures = {
            executor.submit(process_file, file, output_folder): file for file in tokenized_files
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"
        ):
            file, tokens_in_file = future.result()
            logging.info(f"Processed file {file}, total tokens: {tokens_in_file}")
            total_tokens_all += tokens_in_file

    logging.info(f"Total tokens across all files: {total_tokens_all}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Parallel processing using openai/tiktoken to detokenize tokens in tokenized parquet files, tracking progress and token count"
    )
    parser.add_argument(
        "--input_folder", type=str, help="Path to folder containing tokenized parquet files"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Path to output folder for detokenized parquet files"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel processing workers, defaults to CPU core count",
    )
    args = parser.parse_args()
    process_folder_parallel(args.input_folder, args.output_folder, args.num_workers)
