from pathlib import Path

path = Path("../coco128")
fns = list((path/"images/train2017").iterdir())
fns = ['./' + str(fn).split('/',2)[2] for fn in fns]
text = '\n'.join(fns)

with (path/"train2017.txt").open("w") as f:
    f.write(text)

print("coco128 dataset is ready for use")