import openslide
import matplotlib.pyplot as plt
from lxml import etree
import matplotlib.patches as patches

# =========================
# PATHS
# =========================
SLIDE_PATH = "Data/CAMELYON17/images/patient_017_node_2.tif"
XML_PATH = "Data/CAMELYON17/annotations/patient_017_node_2.xml"

# =========================
# FUNCTIONS (from previous scripts)
# =========================
def load_slide(path):
    return openslide.OpenSlide(path)

def load_annotations(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = root.findall(".//Annotation")
    return annotations

def get_bounding_box(annotation):
    coords = annotation.findall(".//Coordinate")
    points = [(float(c.get("X")), float(c.get("Y")))
              for c in coords if c.get("X") and c.get("Y")]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return min(xs), min(ys), max(xs), max(ys)

def show_full_slide_with_bbox(slide, bbox):
    level = slide.level_count - 1
    downsample = slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    img = slide.read_region(
        location=(0, 0),
        level=level,
        size=(width, height)
    ).convert("RGB")

    xmin, ymin, xmax, ymax = bbox
    rect_x = xmin / downsample
    rect_y = ymin / downsample
    rect_w = (xmax - xmin) / downsample
    rect_h = (ymax - ymin) / downsample

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.add_patch(
        patches.Rectangle(
            (rect_x, rect_y),
            rect_w,
            rect_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
    )
    ax.set_title("Vue globale du ganglion avec zone cancéreuse")
    ax.axis("off")
    plt.show()

def show_zoom_on_cancer(slide, bbox, level=2):
    xmin, ymin, xmax, ymax = bbox
    downsample = slide.level_downsamples[level]

    width = int((xmax - xmin) / downsample)
    height = int((ymax - ymin) / downsample)

    region = slide.read_region(
        location=(int(xmin), int(ymin)),
        level=level,
        size=(width, height)
    ).convert("RGB")

    plt.figure(figsize=(6, 6))
    plt.imshow(region)
    plt.title("Zoom sur la région cancéreuse")
    plt.axis("off")
    plt.show()

# =========================
# MAIN PIPELINE
# =========================
slide = load_slide(SLIDE_PATH)
annotations = load_annotations(XML_PATH)

# On prend la première métastase annotée
bbox = get_bounding_box(annotations[0])

show_full_slide_with_bbox(slide, bbox)
show_zoom_on_cancer(slide, bbox)