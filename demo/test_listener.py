from data import Color, Shape, demo_object, all_objects
from run_gpt_prompt import listener_select
import json

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

def test_listener():
    results = []
    total = len(all_objects)

    for target in all_objects:
        # Build learned vocabulary: all objects except target (avoid answer leakage)
        learned_vocabulary = {str(obj): obj_map[str(obj)] for obj in all_objects if obj != target}

        # Get the word for target from ground truth
        word = obj_map[str(target)]

        # Listener selects object given the word
        raw_output, choice_map = listener_select(
            word=word,
            target_object=target,
            learned_vocabulary=learned_vocabulary,
            letters=letters,
            max_word_len=max_word_len,
            model="qwen/qwen3-vl-235b-a22b-instruct"
        )

        # Extract option from listener output JSON
        try:
            parsed = json.loads(raw_output)
            extracted_option = str(parsed.get("option", "")).strip().upper()
        except:
            extracted_option = raw_output.strip().upper()

        selected_obj = choice_map.get(extracted_option)

        # Check if selected object matches target
        is_correct = selected_obj is not None and str(selected_obj) == str(target)
        results.append({
            "target": str(target),
            "word": word,
            "option": extracted_option,
            "selected": str(selected_obj) if selected_obj is not None else "<parse_failed>",
            "correct": is_correct
        })
        print(
            f"Target: {target} | Word: {word} | Option: {extracted_option} | "
            f"Selected: {selected_obj if selected_obj is not None else '<parse_failed>'} | Correct: {is_correct}"
        )

    success_count = sum(1 for r in results if r["correct"])
    success_rate = success_count / total * 100
    print(f"\n=== Results ===")
    print(f"Success: {success_count}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    return results, success_rate

if __name__ == "__main__":
    test_listener()
