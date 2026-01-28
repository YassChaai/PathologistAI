import openslide
import matplotlib.pyplot as plt

# chemin vers une image CAMELYON
SLIDE_PATH = "../Data/CAMELYON17/images/patient_017_node_4.tif"

slide = openslide.OpenSlide(SLIDE_PATH)

print("Nombre de niveaux :", slide.level_count)
print("Dimensions par niveau :", slide.level_dimensions)
print("Downsamples :", slide.level_downsamples)

# on prend le niveau le plus bas (vue globale)
level = 4 ##slide.level_count - 1
width, height = slide.level_dimensions[level]

# on lit toute l'image à ce niveau
img = slide.read_region(
    location=(0, 0),
    level=level,
    size=(width, height)
)

# read_region renvoie une image RGBA
img = img.convert("RGB")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Vue globale - niveau {level}")
plt.axis("off")
plt.show()