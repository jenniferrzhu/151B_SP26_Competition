from datasets import load_dataset
ds = load_dataset("zwhe99/DeepMath-103K", split="train")
print(ds)                  # schema + row count
print(ds[0])               # one row to see field names + types