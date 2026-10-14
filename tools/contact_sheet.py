import sys, glob, os, cv2, numpy as np
out, files = sys.argv[1], sys.argv[2:]
tiles = []
for f in files:
    im = cv2.imread(f)
    im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.putText(im, os.path.basename(f), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(im)
if len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
