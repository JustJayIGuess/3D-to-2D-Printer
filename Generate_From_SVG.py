# Note https://shinao.github.io/PathToPoints/ to get .points data from SVG - .points is a made-up file extension I've decided to use
# numpy = no fun
import math
from svg.path import *
from xml.dom import minidom
import itertools
import numpy as np

import_file = "points.points"

speed = 5000        # Feed rate of gcode
jump_height = 2     # Height that pen will lift between strokes
join_paths = False  # Should all paths be joined? If True, then the pen will not be lifted
keep_aspect = True  # Keep aspect ratio when transforming to bounds? If True, the drawing will be fit within the bounds
num_samples = 5     # Number of points to sample on non-linear curves of SVG

set_bounds = np.array([[0, 42], [194, 220]]) # The bounds for the drawing to fit into. Note that the drawing will be centered in these bounds
scale_from_bounds = 0.5             # The scale of the drawing within the bounds. If 1.0, then the drawing will be the maximum size that fits in bounds

paths = []  # Stores all strokes [stroke1, stroke2, etc] where stroke1 = [point1, point2, etc]

answer_dict = {"y": True, "yes": True, "n": False, "no": False}



def create_grid_matrix(side_count: int):
    grid = np.zeros((side_count ** 2, side_count ** 2))
    for i in range(side_count ** 2 - 1):
        grid[i][i+1] = int((i % side_count) != (side_count - 1))
        grid[i+1][i] = grid[i][i+1]
    for i in range(side_count ** 2 - side_count):
        grid[i][i+side_count] = 1
        grid[i+side_count][i] = 1
    print(grid)

create_grid_matrix(4)


print("Use https://shinao.github.io/PathToPoints/ to convert SVG to a points list compatible with this program.")
import_file = input("Import File: ")
speed = round(float(input("Speed (mm/s): ")) * 60)
# jump_height = float(input("Lift Height (3 usually works well): "))
# join_paths = answer_dict[input("Join Disconnected Paths (y/n): ")]
num_samples = int(input("Samples Per Nonlinear Curve: "))
# bounds_raw = input("Image Bounds (e.g., 0 0 10 15): ")
# set_bounds = [[float(bounds_raw.split(" ")[2*i+j]) for j in range(2)] for i in range(2)]
# keep_aspect = answer_dict[input("Keep Aspect Ratio (y/n): ")]
scale_from_bounds = min(abs(float(input("Scale (0-1): "))), 1)


# Note: path_arr must be a nice rectangular table - no jagged funkiness
# Also im proud of this function, it looks pretty
def get_path_bounds(arr):
    #arr = list(itertools.chain.from_iterable(arr))
    min_bound = np.amin(arr, axis=0) # [min(arr, key = lambda p: p[i])[i] for i in range(2)]
    max_bound = np.amax(arr, axis=0) # [max(arr, key = lambda p: p[i])[i] for i in range(2)]
    return np.array([min_bound, max_bound])

def bounds_center(bounds):
    return (bounds[0] + bounds[1]) / 2.0

# Get the scale and offset to apply to the imported paths to achieve the target bounds
def get_bounds_transform(original, image, do_keep_aspect=False):
    scale = (image[1] - image[0]) / (original[1] - original[0])
    if do_keep_aspect:
        scale = np.array([min(scale), min(scale)])
    scale *= scale_from_bounds
        
    offset = image[0] - scale * original[0]
    if do_keep_aspect:
        new_bounds = np.array([[original[i][j] * scale[j] + offset[j] for j in range(2)] for i in range(2)])
        offset = offset + (image[1] - new_bounds[1]) / 2
        
    return scale, offset

# Transform all paths according to scale and offset
def transform_paths(scale, offset):
    global paths
    paths = [scale * path + offset for path in paths]

def sqr_dist(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

print("\nImporting and parsing file...")
doc = minidom.parse(import_file)  # parseString also exists
path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
path_transforms = [path.getAttribute('transform') for path in doc.getElementsByTagName('path')]
for i, t in enumerate(path_transforms):
    if t == "":
        path_transforms[i] = np.array([0, 0])
    else:
        path_transforms[i] = np.array([float(t.split(", ")[0][10:]), float(t.split(", ")[1][:-1])])

doc.unlink()
paths_data = []
for path_string in path_strings:
    paths_data.append(parse_path(path_string))

# Flatten the list
acc = 0 # Accumulator
end_tag_indices = []
for paths_datum in paths_data:
    acc += len(paths_datum)
    end_tag_indices.append(acc)
paths_data = np.concatenate(paths_data)

# Convert to paths 3d array format and split based on tags to apply tag-wide transforms
temp_path = []
move_vert_indices = []
path_start = paths_data[0]
vert_increment = 0
for i, paths_datum in enumerate(paths_data):
    if i == end_tag_indices[0]:
        temp_path = np.array(temp_path) + path_transforms[0]
        path_transforms.pop(0)
        paths.append(np.array(temp_path))
        temp_path = []
        end_tag_indices.pop(0)
    if isinstance(paths_datum, Move):
        temp_path.append([paths_datum.start.real, paths_datum.start.imag])
        path_start = [paths_datum.start.real, paths_datum.start.imag]
        vert_increment += 1
        if i != 0:
            move_vert_indices.append(vert_increment)
    elif isinstance(paths_datum, Line):
        temp_path.append([paths_datum.end.real, paths_datum.end.imag])
        vert_increment += 1
    elif isinstance(paths_datum, Close):
        temp_path.append(path_start)
        vert_increment += 1
    else:
        for t in range(num_samples):
            sample = paths_datum.point(t / num_samples)
            temp_path.append([sample.real, sample.imag])
        vert_increment += num_samples

temp_path = np.array(temp_path) + path_transforms[0]
paths.append(np.array(temp_path))

# Re-split paths at Move instructions for path optimisation later
split_paths = []
i = 0
temp_path = []
for path in paths:
    for vert in path:
        if i + 1 in move_vert_indices:
            split_paths.append(np.array(temp_path))
            temp_path = []
        temp_path.append(vert)
        i += 1

split_paths.append(np.array(temp_path))
paths_ = paths.copy()
paths = split_paths

# Flip and transform imported points into specified bounds
print("Transforming points...")
for i in range(len(paths)):
    paths[i] *= np.array([1, -1])
scale, offset = get_bounds_transform(get_path_bounds(np.concatenate(paths)), set_bounds, keep_aspect)
transform_paths(scale, offset)

# Reorder paths to minimise travel time - just uses bounding boxes as a rough approximation
print("Calculating path centers...")
centers = []
for path in paths:
    centers.append(bounds_center(get_path_bounds(path)))

dist_mat = [[0] * len(centers) for _ in range(len(centers))]

#   Generate distance matrix
total_progress = (len(centers) - 1) * (len(centers) - 2) // 2
update_frequency = total_progress / 20
iter = 0
for i in range(len(centers) - 1):
    for j in range(i + 1, len(centers)):
        if iter % update_frequency < 1:
            n = round(iter / update_frequency)
            print("Computing distance matrix: " + "#" * n + "-" * (20 - n), end="\r")
        dist_mat[i][j] = sqr_dist(centers[i], centers[j])
        dist_mat[j][i] = dist_mat[i][j]
        iter += 1
        
print("Computing distance matrix: " + "#" * 20)
del iter

#   Sort using greedy (bc im lazy) based on matrix
sorted_indices = [0]
unsorted_indices = list(range(1, len(centers)))
total_progress = len(centers) - 1
update_frequency = total_progress / 20
for iter in range(len(centers) - 1):
    if iter % update_frequency < 1:
        n = round(iter / update_frequency)
        print("Sorting: " + "#" * n + "-" * (20 - n), end="\r")

    last = sorted_indices[-1]
    selected = unsorted_indices[0]
    min_dist = dist_mat[selected][last]
    for i in unsorted_indices[1:]:
        if dist_mat[i][last] < min_dist:
            selected = i
            min_dist = dist_mat[i][last]

    sorted_indices.append(selected)
    unsorted_indices.remove(selected)

print("Sorting: " + "#" * 20)
sorted_paths = []
for i in sorted_indices:
    sorted_paths.append(paths[i])
paths = sorted_paths

# Add the initialising gcode commands
print("Constructing file...")
res = ""
with open("header.txt", "r") as file:
    for line in file:
        res += line

res += "\n"

# Convert strokes to movement commands between points and pen lifts as needed
for path in paths:
    res += "G0 Z" + str(jump_height) + " F" + str(round(speed)) + "\n"
    res += "G0 X" + str(path[0][0]) + " Y" + str(path[0][1]) + "\n"
    res += "G0 Z0.2\n"
    for vert in path[1:]:
        res += "G0 X" + str(vert[0]) + " Y" + str(vert[1])
        res += "\n"

# Add finalising gcode
with open("footer.txt", "r") as file:
    for line in file:
        res += line

# Ask for filename and export the gcode as a file
filename = input("Filename (w/o .gcode; leave blank to cancel): ")
if not filename == "":
    print("Writing to file...")
    with open(filename + ".gcode", "w") as file:
        file.write(res)
    
    input("Done! Press any key to exit.")
else:
    print("Cancelled.")
