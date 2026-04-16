from utils.brain import retrieve_context
ctx = retrieve_context("Ap ka pass kon kon se products ha?", top_k=15)
lines = ctx.split("\n")
for i, line in enumerate(lines):
    try:
        print(f"[{i}] {line}")
    except:
        print(f"[{i}] (unicode error)")
