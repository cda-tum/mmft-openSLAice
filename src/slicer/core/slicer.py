from src.slicer.core.stl import Stl
from src.slicer.core.printer import Printer
from src.slicer.core.resin import Resin
from PIL import Image
import src.slicer.file_types
import math
import numpy as np
from enum import Enum, auto
from src.slicer.core.section import FeatureSection, Feature
from src.slicer.core.constants import *
from src.slicer.core.helper_functions import round_TOL
from src.slicer.core.file_type import FileTypeRegistry






# enum for how the intersection levels, orientation, placement are computed
class ParameterMode(Enum):
    AUTO = auto()
    MANUAL = auto()



class Slicer:
    """    
    A class to hold all stl files, a printer object, and a resin object and the functions for the slicing process.

    Attributes
    ----------
    printer: Printer
        Printer object
    resin: Resin
        Resin object
    stls: list[Stl]
        List of Stl objects to be sliced
    layer_height_method: ParameterMode
        How the layer heights are computed (AUTO or MANUAL)
    forced_layer_height: float
        The layer height in mm to be used if layer_height_method is MANUAL
    feature_sections: list[FeatureSection]
        List of FeatureSection objects defining the sections of the print
    intersection_levels: list[float]
        List of intersection levels (z-positions in mm) where the stls are sliced
    layers_image: list[Image.Image]
        List of PIL Image objects representing the rasterized layers of the print
    """
    
    def __init__(self, printer: Printer, resin: Resin):
        self.stls: list[Stl] = []
        self.printer: Printer = printer
        self.resin: Resin = resin
        self.layer_height_method: ParameterMode = ParameterMode.AUTO # defaults to auto
        self.forced_layer_height: float = 0.05 # defaults to 50um
        self.feature_sections: list[FeatureSection] = []
        self.intersection_levels: list[float] = []
        self.layers_image: list[Image.Image] = []


# getter & setter methods

    def add_input_file(self, file_path: str) -> None:
        self.stls.append(Stl(file_path, self)) 

    def get_stls(self) -> list[Stl]:
        return self.stls

    def set_printer(self, printer: Printer) -> None:
        self.printer = printer

    def set_resin(self, resin: Resin) -> None:
        self.resin = resin

    def set_layer_height_method(self, layer_height_method: ParameterMode) -> None:
        self.layer_height_method = layer_height_method

    def get_layer_height_method(self) -> ParameterMode:
        return self.layer_height_method

    def set_forced_layer_height(self, forced_layer_height: float) -> None:
        self.forced_layer_height = forced_layer_height

    def get_forced_layer_height(self) -> float:
        return self.forced_layer_height
    
    def get_layer_heights(self) -> list[float]:
        """Get the layer heights of all the layers"""
        z_h = []
        prev_z = 0.0
        for z in self.intersection_levels:
            z_h.append(round_TOL((z-prev_z), self.printer.settings.z_resolution))
            prev_z = z
        return z_h

    def get_volume(self) -> float:
        """Calculate the volume of all stls in ml"""
        volume = 0.0
        for stl in self.stls:
            volume += stl.get_volume()
        return volume


# slicing methods

    def auto_orientation(self) -> bool:
        """
        Automatically orient all stls to the best orientation for printing.
        """
        for stl in self.stls:
            stl.auto_orientation()
        return True


    def auto_arrange(self, distance: float) -> bool:
        """
        Automatically arrange the mesh in the xy plane. All the stls will be arranged in a grid pattern
        with a fixed distance between them. The distance is defined in the parameters.
        Note: very simple implementation!

        Parameters
        ----------
        distance : float
            The distance [mm] between the stls in the xy plane.
        Returns
        -------
        bool
            True if the arrangement was successful, False otherwise.
        """
        # get printer dimensions
        printer_width, printer_depth = self.printer.settings.screen_resolution
        distance = math.ceil(distance / self.printer.settings.pixel_size)  # Convert to pixels

        # Calculate bounding boxes for each STL
        bounding_boxes = []
        for stl in self.stls:
            bounding_boxes.append(stl.prepare_arrange())

        n_stls = len(self.stls)
        if n_stls == 0:
            return False

        # extract the optimized dimensions  
        max_width = max(box[0] for box in bounding_boxes)
        max_depth = max(box[1] for box in bounding_boxes)

        # calculate desired grid dimensions by approximating the printer aspect ratio   
        printer_aspect = printer_width / printer_depth
        columns = max(1, round(math.sqrt(n_stls * printer_aspect)))
        rows = math.ceil(n_stls / columns)

        # compute total area required for the grid (including margins)
        grid_cell_width = max_width
        grid_cell_depth = max_depth
        total_width = columns * grid_cell_width + (columns + 1) * distance
        total_depth = rows * grid_cell_depth + (rows + 1) * distance

        # check if the grid fits within the printer's area
        if total_width > printer_width or total_depth > printer_depth:   
            return False

        # center the grid within the printer's area
        x_offset = math.ceil((printer_width - total_width) / 2.0)   
        y_offset = math.ceil((printer_depth - total_depth) / 2.0)

        # arrange the STLs  
        for i, stl in enumerate(self.stls):
            col = i % columns 
            row = i // columns
            x = x_offset + distance + col * (grid_cell_width + distance)
            y = y_offset + distance + row * (grid_cell_depth + distance)
            stl.set_position(x, y)
        return True


    def _compute_feature_sections(self) -> bool:
        """
        Compute the feature sections based on the z-levels and top/bottom bounds of the stls.
        It will also ensure that the z-levels are spaced according to the minimum layer height and the printer's z-resolution.
        """
        def enforce_min_distance(numbers: list[float], start: float, end: float, min_d: float, z_res: float) -> list[float]:
            """
            Ensure that the numbers are spaced at least min_d apart, starting from start and ending at end.
            If the numbers are too close together, they will be adjusted to ensure the minimum distance (if possible).

            Arguments:
            ----------
            numbers : list[float]
                The list of numbers to adjust.
            start : float
                The start of the range.
            end : float
                The end of the range.
            min_d : float
                The minimum distance between numbers.
            z_res : float
                The z-resolution to use for rounding.

            Returns
            -------
            list[float]
                The adjusted list of numbers.
            """
            # prepare input data: truncate numbers to the range [start, end] & ensure start and end are present
            numbers = [num for num in numbers if start <= num <= end]
            if start not in numbers:
                numbers.insert(0, start)
            if end not in numbers:
                numbers.append(end)
            # select numbers ensuring min_d spacing
            result = [start]
            last = start
            prev_last = None
            for i, num in enumerate(numbers):
                if num == start or num == end:
                    continue  # already handled
                if num - last >= min_d:
                    # if current section would be big enough -> add value
                    result.append(num)
                    prev_last = result[-2]
                    last = num
                elif prev_last and prev_last - num >= 2* min_d:
                    # current section too small but when shifting last value to the left (to the last_last is enough space to do so) the section is big enough
                    result[-1] = round_TOL(num - min_d, z_res)
                    result.append(num)
                elif len(numbers) > i + 1 and numbers[i + 1] - last >= 2 * min_d:
                    # current section too small but when shifting the current value to the right (to the nect is enough space to do so) the section is big enough
                    result.append(round_TOL(last + min_d, z_res))
                    last = last + min_d

            # Always include end
            if end - result[-1] >= min_d or result[-1] == end:
                result.append(end)
            else:
                if prev_last and prev_last - end >= 2* min_d:
                    result[-1] = round_TOL(end - min_d, z_res)
                # Remove last one if too close to end
                else:
                    result.pop()
                result.append(end)
            return result

        # collect unique z-levels and top positions
        combined_z_levels = set()
        combined_top_pos = set()
        combined_bottom_pos = set()
        for stl in self.stls:
            combined_z_levels.update(stl.get_z_feature_levels())
            combined_top_pos.update(stl.get_top_layer_pos())
            combined_bottom_pos.update(stl.get_bottom_layer_pos())

        sorted_z_levels: list[float] = sorted(combined_z_levels)
        print (f"sorted_z_levels: {sorted_z_levels}")
        min_z = sorted_z_levels[0]
        max_z = sorted_z_levels[-1]
        sorted_top_bottom_pos = sorted(combined_top_pos | combined_bottom_pos)

        # enforce spacing on top layers to define initial major z-positions
        initial_final_z_levels: list[float] = enforce_min_distance(sorted_top_bottom_pos, min_z, max_z, self.printer.min_layer_height, self.printer.settings.z_resolution)
        print (f"initial_final_z_levels: {initial_final_z_levels}")

        # enrich intervals between each pair in initial_final_z_levels
        additional_z_levels = set()
        for i in range(len(initial_final_z_levels) - 1):
            a, b = initial_final_z_levels[i], initial_final_z_levels[i + 1]
            print (f"enriching between {a} and {b}")
            enriched = enforce_min_distance(sorted_z_levels, a, b, self.printer.min_layer_height, self.printer.settings.z_resolution)
            print (f"-> enriched: {enriched}")
            additional_z_levels.update(enriched[1:-1])  # skip a and b

        # Merge and sort final z-levels
        print (f"additional_z_levels: {additional_z_levels}")
        final_z_levels = sorted(set(initial_final_z_levels) | additional_z_levels)
        dropped_z_levels = set(combined_z_levels) - set(final_z_levels)
        print(f"final_z_levels: {final_z_levels}")

        # => create FeatureSections
        sections = FeatureSection.from_boundaries(final_z_levels)
        for section in sections:
            # add TL feature if section starts at a top layer value
            if section.start in combined_top_pos:
                section.add_feature(Feature.TL)
            # Add F if any dropped z-level lies within this section
            if any(section.includes_point(z) for z in dropped_z_levels):
                section.add_feature(Feature.F)
            for stl in self.stls:
                # add MF feature if it overlaps with one of the mf_sections of the stls
                if Feature.MF not in section.features and stl.mf_section.overlaps_with(section):
                    section.add_feature(Feature.MF)
                # set max angle (most dominant)
                angle = stl.find_dominant_angle(section)
                if angle > section.angle:
                    section.set_angle(angle)
        
        self.feature_sections = sections
        return True


    def compute_intersection_levels(self) -> list[float]:
        """
        Compute the intersection levels for all stls based on the layer height method.
        If the layer height method is set to MANUAL, it will use the forced layer height.
        If the layer height method is set to AUTO, it will compute the intersection levels based on the feature sections.
        Returns:
            list[float]: A list of intersection levels (z-positions in mm) where the stls are sliced.
        """
        z_res = self.printer.settings.z_resolution
        min_lh = self.printer.min_layer_height
        max_lh = self.printer.max_layer_height

        # if forced layer height is set, use it for the whole z range
        if self.layer_height_method == ParameterMode.MANUAL:
            if self.forced_layer_height < min_lh:
                raise ValueError(f"Forced layer height {self.forced_layer_height} is smaller than the minimum layer height {min_lh}.")
            if self.forced_layer_height > max_lh:
                raise ValueError(f"Forced layer height {self.forced_layer_height} is bigger than the maximum layer height {max_lh}.")
            z_max = max([stl.get_z_max() for stl in self.stls])
            self.intersection_levels = np.arange(self.forced_layer_height, z_max + self.forced_layer_height, self.forced_layer_height).tolist()
            self.intersection_levels = [round_TOL(z, z_res) for z in self.intersection_levels]
            return self.intersection_levels
        
        # if auto layer height is set, use feature sections to compute the layer heights
        self._compute_feature_sections()
        intersection_levels = set()
        for section in self.feature_sections:
            intersection_levels.update(section.get_intersection_levels(min_lh=min_lh, max_lh=max_lh, z_res=z_res))
            
        self.intersection_levels = sorted(intersection_levels)[1:] # remove first intersection at 0.0 (not possible)
        print(f"intersection_levels: {self.intersection_levels}")
        return self.intersection_levels


    def slice_all(self) -> bool:
        """
        Slice all stls based on the intersection levels.
        """
        if not self.intersection_levels:
            self.compute_intersection_levels()
        for stl in self.stls:
            stl.slice(self.intersection_levels)
        return True


    def rasterize(self, min_aa: int) -> bool:
        """
        Rasterize all stls for each intersection level.
        """
        for i, stl in enumerate(self.stls):
            print(f"Rasterizing stl-{i} ...")
            stl.rasterize(min_aa)
        self.layers_image = []
        for i in range(len(self.intersection_levels)):
            combined_layer = Image.new("L", self.printer.settings.screen_resolution, color=0)
            for stl in self.stls:
                rastericed_layer = stl.get_rasterized_layer_i(i)
                if rastericed_layer is not None:
                    combined_layer.paste(rastericed_layer, stl.get_position())
            #combined_layer.save(f"/home/benni/Documents/Repos/MF-SLA-Slicer/src/slicer/build/temp_{i}.png")
            self.layers_image.append(combined_layer)
        print(f"Rasterized {len(self.layers_image)} layers.")
        return True


    def save(self, file_name: str, save_path: str) -> bool:
        """
        Save the slicing result using the correct file type.

        Args:
            file_name: Name of the file to be saved
            save_path: Path where the file should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        print(f"layer heights: {self.get_layer_heights()}")
        print(f"Saving file {file_name} to {save_path}")
        file_type_class = FileTypeRegistry.get(self.printer.export_file_type)  # Get the file type class from the registry#
        file_type_object = file_type_class(
            file_name, self.printer, self.resin,
            self.layers_image, self.intersection_levels,
            self.get_layer_heights(), self.get_volume()
        )
        return file_type_object.save(save_path)

