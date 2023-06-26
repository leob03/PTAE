print("First let us run the pretrained encoder on the test dataset of ModelNet40 and evaluate if we obtain the same results as on the report")
print("\n")
print("we should obtain a Test Instance Accuracy around 82% and a Class Accuracy around 73%")

with open("eval_classification.py") as f:
    exec(f.read())

print("\n")
print("Now let us run the whole pretrained network on a handful of objects from the test dataset of ShapetNetPart and evaluate if we obtain the same results as on the report")
print("\n")
print("we should obtain Class Average mIOU around 80% and an Inctance Average mIOU also around 80%")


with open("eval_partsegmentation.py") as f:
    exec(f.read())