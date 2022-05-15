# IMPORTS
import sys
import math
import timeit
import subprocess
import pkg_resources

# INSTALL MISSING PACKAGES
required = {"opencv-contrib-python", "numpy"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("missing the following packages: " + ", ".join(missing) + "...")
    print("installing...")
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    print("finished installing...")

import cv2
import numpy as np

from line_utils import merge_lines, intersection, is_on_line

# CONSTANTS
merge_dist = 30
timeit.default_timer()

# CLASSES
class Polygon():
    def __init__(self, initLine):
        self.lines = []
        self.pts = [*initLine]

    @property
    def perimeter(self):
        return sum(map(lambda line: math.dist(line[0], line[1]), self.lines))

    def get_pts(self):
        pts = []
        for pt in self.pts:
            if pt not in pts:
                pts.append(pt)

        return pts

    def add(self, line):
        if line in self.lines:
            return False

        can_add = False
        for pt in line:
            if pt in self.pts:
                can_add = True
                break

        if can_add:
            self.pts += [*line]
            self.lines.append(line)

        return can_add

# Poll
if len(sys.argv) < 2:
    print("Please drag and drop your image on top of scan.py")
    input("Press enter to exit... ")
    exit()

# Load Image
img_color = cv2.imread(sys.argv[1])
try:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
except:
    print("Image '%s' does not exist, or is not an image" % sys.argv[1])
    input("Press enter to exit... ")
    exit()

# Blur image without ruining edges
img_blur = cv2.bilateralFilter(img_gray, 15, 75, 75)

# Highlight edges
img_edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=100)

# Close gaps
kernel = np.ones((5,5),np.uint8)
img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# Get lines from highlighted edges
edges = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi/180, threshold=50,
                        minLineLength=0, maxLineGap=20)

lines = []
for [[x0, y0, x1, y1]] in edges:
    lines.append([(x0, y0), (x1, y1)])

# Merge lines by distance & angle
lines = merge_lines(lines, max_distance_to_merge=merge_dist, max_angle_to_merge=6.5)

# Merge lines via Projection
for a, lineA in enumerate(lines):
    for b, lineB in enumerate(lines):
        if a == b:
            continue

        all_pts = [*lineA, *lineB]
        if len(all_pts) != len(set(all_pts)):
            continue
        
        intersection_point = intersection(lineA, lineB)
        if intersection_point == None or any(val < 0 for val in intersection_point):
            continue
        
        _merge_dist = merge_dist if not is_on_line(intersection_point, lineA) else 3
        dists = {}

        for pt in [*lineA, *lineB]:
            if not dists.get(pt):
                dists[pt] = math.dist(pt, intersection_point)

        min_dist = min(map(lambda pt: dists[pt], [*lineA, *lineB]))

        if min_dist <= _merge_dist:
            def connect(line):
                if dists[line[0]] < dists[line[1]]:
                    return [intersection_point, line[1]]
                else:
                    return [line[0], intersection_point]

            lines[a] = connect(lineA)
            lines[b] = connect(lineB)

# Generate Polygons
polygons = []
changes_made = True
lines_added = 0

while changes_made and lines_added < len(lines):
    changes_made = False

    for line in lines:
        for poly in polygons:
            added = poly.add(line)
            
            if added:
                lines_added += 1
                changes_made = True
                break

    if not changes_made and lines_added != len(lines):
        polygons.append(Polygon(lines[lines_added]))
        changes_made = True
        lines_added += 1

# Get Quadrilaterals
quads: list[Polygon] = list(filter(lambda poly: len(poly.lines) == 4, polygons))
if not quads:
    print("No paper detected")
    input("Press enter to exit... ")
    exit()

# TODO: get quad based on parallelism... area if parallelism too similar
quad = max(quads, key=lambda poly: poly.perimeter)

# Identify corners of quad
pts = quad.get_pts()
x_sorted = list(sorted(pts, key=lambda pt: pt[0]))
y_sorted = list(sorted(pts, key=lambda pt: pt[1]))
top = y_sorted[:2]


tl = sorted(top, key=lambda pt: pt[0])[0]
tr = top[1] if tl == top[0] else top[0]
x_sorted.remove(tl)
x_sorted.remove(tr)

bl = x_sorted[0]
br = x_sorted[1]

# Generate warp matrix from detected points
width = int(max(math.dist(tl, tr), math.dist(bl, br)))
height = int(max(math.dist(tl, bl), math.dist(tr, br)))

def convertToPoints(pts):
    return np.float32(np.array(pts)[:, np.newaxis, :])

srcPoints = convertToPoints([tl, tr, bl, br])
dstPoints = convertToPoints([(0, 0), (width, 0), (0, height), (width, height)])

homography, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)

# Warp original image & Apply Filters
transformed_img = cv2.warpPerspective(img_color, homography, (width, height))
transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)

# transformed_img = cv2.medianBlur(transformed_img, 3)
# transformed_img = cv2.bilateralFilter(transformed_img, 5, 10, 10)
# transformed_img = cv2.adaptiveThreshold(transformed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21, 4)

# Finalize
print("RUN TIME:", timeit.default_timer())
cv2.imwrite("out.png", cv2.cvtColor(transformed_img, cv2.COLOR_GRAY2RGB))