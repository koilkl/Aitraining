import os


application = os.path.join("dist", "TFLiteTraining.app")
files = [application]
symlinks = {"Applications": "/Applications"}

icon = None
background = None

format = "UDZO"
size = "480M"

window_rect = ((200, 200), (700, 420))
icon_locations = {
    "TFLiteTraining.app": (140, 220),
    "Applications": (520, 220),
}

