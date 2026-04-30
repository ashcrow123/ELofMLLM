from model import all_images, image
from llm_provider.run_gpt_prompt import speaker_retrieval_async
import asyncio
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List, Set
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-4o-mini")
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--batch", type=int, default=9)
args = parser.parse_args()

async def retrieve_one(
    idx: int,
    shuffled_images: List[image],
    model: str,
    semaphore: asyncio.Semaphore,
    batch_size: int,
):
    candidate_images = shuffled_images[idx + 1:]
    if not candidate_images:
        return None

    target_image = shuffled_images[idx]
    selected_nums: Set[int] = set()
    for batch_start in range(0, len(candidate_images), batch_size):
        batch_images = candidate_images[batch_start:batch_start + batch_size]
        if not batch_images:
            continue

        async with semaphore:
            try:
                output = await speaker_retrieval_async(
                    target_image=target_image,
                    all_images=batch_images,
                    model=model,
                )
            except Exception as e:
                raise RuntimeError(
                    f"target={target_image.num}, "
                    f"batch_start={batch_start}, "
                    f"batch_size={len(batch_images)}, "
                    f"candidates={len(candidate_images)}, "
                    f"error={e}"
                ) from e

        output = json.loads(output)
        for num in output["num_list"]:
            if not isinstance(num, int) or num < 0 or num >= len(batch_images):
                continue

            selected_num = batch_images[num].num
            if target_image.num != selected_num:
                selected_nums.add(selected_num)

    return target_image, selected_nums


async def main():
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")

    results: Dict[str, Set[int]] = {str(im.num): set() for im in all_images}
    model = args.model

    shuffled_images: List[image] = list(all_images)
    random.shuffle(shuffled_images)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        retrieve_one(i, shuffled_images, model, semaphore, args.batch)
        for i in range(len(shuffled_images) - 1)
    ]
    failed_targets: List[str] = []

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            item = await task
        except Exception as e:
            failed_targets.append(str(e))
            continue

        if item is None:
            continue

        target_image, selected_nums = item
        for selected_num in selected_nums:
            target_num = target_image.num

            results[str(target_num)].add(selected_num)
            results[str(selected_num)].add(target_num)

    serializable_results: Dict[str, List[int]] = {
        image_num: sorted(nearby_nums)
        for image_num, nearby_nums in results.items()
    }

    output_dir = Path("./object_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model.replace('/', '-')}_network.json", "w") as f:
        json.dump(serializable_results, f, indent=4)

    if failed_targets:
        failed_path = output_dir / f"{model.replace('/', '-')}_failed.json"
        with open(failed_path, "w") as f:
            json.dump(failed_targets, f, indent=4)
        print(f"{len(failed_targets)} retrieval tasks failed. See {failed_path}")


if __name__ == "__main__":
    asyncio.run(main())
