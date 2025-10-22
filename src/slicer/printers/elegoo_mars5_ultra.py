from dataclasses import dataclass
from ..core.printer import Printer, PrinterSettings, PrinterRegistry



settings = PrinterSettings(
    screen_resolution=(8520, 4320),  # 9K resolution: 8520x4320 pixels
    pixel_size=0.018,  # 18 μm = 0.018 mm
    max_z_height=165.0,  # 165 mm build height
    z_resolution=0.001,
    max_z_speed=150.0,
    max_z_acceleration=1000.0
)

printer = Printer(
    name="Elegoo Mars 5 Ultra",
    export_file_type="goo",  # Uses .goo file format
    settings=settings,
    min_layer_height=0.01,
    max_layer_height=0.2 
)

PrinterRegistry.register(printer)
