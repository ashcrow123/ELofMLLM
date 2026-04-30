from data import Color, Shape, demo_object, all_objects
from run_gpt_prompt import speaker_generate

# Ground truth vocabulary
color_map = {
    "red": "ap",
    "green": "be",
    "blue": "ki"
}
shape_map = {
    "cube": "lo",
    "sphere": "zu",
    "tetrahedron": "mi"
}

obj_map = {}
for obj in all_objects:
    obj_map[str(obj)] = f"{shape_map[obj.shape.value]}-{color_map[obj.color.value]}"

# Letters (atomic symbols) for the language
letters = ["lo", "zu", "mi", "ap", "be", "ki"]
max_word_len = 10

def test_speaker():
    results = []
    total = len(all_objects)

    for target in all_objects:
        # Build learned vocabulary: all objects except target
        learned_vocabulary = {str(obj): obj_map[str(obj)] for obj in all_objects if obj != target}

        # Speaker generates a word for target
        generated_word = speaker_generate(
            target_object=target,
            learned_vocabulary=learned_vocabulary,
            letters=letters,
            max_word_len=max_word_len,
            model="qwen/qwen3-vl-235b-a22b-instruct"
        )

        # Extract word from generated output (handle JSON format)
        ground_truth = obj_map[str(target)]
        try:
            import json
            parsed = json.loads(generated_word)
            extracted_word = parsed.get("word", "").strip()
        except:
            extracted_word = generated_word.strip()

        # Check if word matches ground truth
        is_correct = extracted_word.upper() == ground_truth.upper()
        results.append({
            "target": str(target),
            "ground_truth": ground_truth,
            "generated": extracted_word,
            "correct": is_correct
        })
        print(f"Target: {target} | Truth: {ground_truth} | Generated: {extracted_word} | Correct: {is_correct}")

    success_count = sum(1 for r in results if r["correct"])
    success_rate = success_count / total * 100
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    return results, success_rate

if __name__ == "__main__":
    test_speaker()
