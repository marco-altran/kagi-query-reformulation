from typing import List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_ID = "prhegde/t5-query-reformulation-RL"
# MODEL_ID = "alexdong/query-reformulation-knowledge-base-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()


def rewrite_query(input_sequence: str, nsent: int = 3):
    input_ids = tokenizer(input_sequence, return_tensors="pt").input_ids
    print(f'Input: {input_sequence}')
    with torch.no_grad():
        for i in range(nsent):
            output = model.generate(input_ids, max_length=35, num_beams=1, do_sample=True, repetition_penalty=1.8)
            target_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Target: {target_sequence}')
    print("\n")

@torch.inference_mode()
def reformulate(question: str, n: int = 3) -> List[str]:
    print(f'Input: {question}')
    prompt = question.strip()

    inputs = tokenizer(prompt, return_tensors="pt")

    # Beam search gives cleaner, more focused outputs
    num_beams = max(5 * n, 6)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        num_beams=num_beams,
        num_beam_groups=n,
        num_return_sequences=n,
        diversity_penalty=0.5,
        length_penalty=0.8,
        early_stopping=True,
    )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Each sequence may contain multiple lines; collect until we have n unique queries
    queries, seen = [], set()
    for seq in decoded:
        for line in seq.split("\n"):
            q = line.strip(" -•*\t").strip()
            if q and q.lower() not in seen:
                queries.append(q)
                seen.add(q.lower())
            if len(queries) >= n:
                break
        if len(queries) >= n:
            break

    targets = queries[:n] if queries else decoded[:n]
    for target in targets:
        print(f'Target: {target}')
    print("\n")
    return targets

print("Original\n")
rewrite_query("In what year was the winner of the 44th edition of the Miss World competition born?")
rewrite_query("Who lived longer, Nikola Tesla or Milutin Milankovic?")
rewrite_query("Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?")
rewrite_query("Create a table for top noise cancelling headphones that are not expensive")
rewrite_query("What are some ways to do fast query reformulation")
print("Beam search\n")
reformulate("In what year was the winner of the 44th edition of the Miss World competition born?")
reformulate("Who lived longer, Nikola Tesla or Milutin Milankovic?")
reformulate("Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?")
reformulate("Create a table for top noise cancelling headphones that are not expensive")
reformulate("What are some ways to do fast query reformulation")
