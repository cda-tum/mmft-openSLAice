import numpy as np
import os
import struct
from typing import List
from datetime import datetime
from PIL import Image, ImageFont, ImageDraw
from src.slicer.core.printer import Printer
from src.slicer.core.resin import Resin
from src.slicer.core.file_type import FileType, FileTypeRegistry



class GooFile(FileType):
    """
    Implementation of Elegoo's .goo file format based on the official specification v1.2
    
    The .goo format consists of:
    - Header (0x2FB95 bytes = 195477 bytes, containing all print settings and embedded preview images)
    - Layer definitions (24 bytes per layer with individual settings)
    - Layer image data (RLE encoded)
    
    Based on the Rust implementation from https://github.com/connorslade/goo/ (MIT License)
    All values are stored in BIG-ENDIAN format
    """
    
    type_name: str = "goo"
    
    # Header size constant from Rust implementation
    HEADER_SIZE = 0x2FB95  # 195477 bytes
    
    # Magic constants from GOO specification
    ENDING_STRING = bytes([0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x44, 0x4C, 0x50, 0x00])
    MAGIC_TAG = bytes([0x07, 0x00, 0x00, 0x00, 0x44, 0x4C, 0x50, 0x00])
    DELIMITER = bytes([0x0D, 0x0A])
    
    def __init__(self, file_name: str, printer: Printer, resin: Resin, 
                 layers: List[Image.Image], intersection_levels: List[float],
                 layer_heights: List[float], volume: float):
        """
        Initialize the Goo file with print parameters
        
        Parameters
        ----------
        file_name : str
            Name of the file (without extension)
        printer : Printer
            Printer configuration object
        resin : Resin
            Resin configuration object
        layers : List[Image.Image]
            List of layer images (grayscale)
        intersection_levels : List[float]
            Z-height at which each layer starts
        layer_heights : List[float]
            Thickness of each layer
        volume : float
            Total print volume in mm³
        """
        self.file_name = file_name
        self._printer = printer
        self._resin = resin
        self._layers = layers
        self._intersection_levels = intersection_levels
        self._layer_heights = layer_heights
        self._volume = volume
        
        # Prepare layer data with individual settings
        self._layer_data = self._prepare_layer_data()
        

    def _prepare_layer_data(self) -> List[dict]:
        """
        Prepare layer data with individual exposure and movement settings
        
        Returns
        -------
        List[dict]
            List of dictionaries containing encoded layer data and settings
        """
        layer_data = []
        total_layers = len(self._layers)
        print(f"Encoding {total_layers} layers for .goo file...")
        
        for i, layer in enumerate(self._layers):
            img_array = np.array(layer)  # Convert PIL Image to numpy array
            encoded_data, checksum = self._encode_layer_rle(img_array)
            exposure_time = self._resin.get_exp_time(i, self._layer_heights[i])
            layer_info = {
                'data': encoded_data,
                'checksum': checksum,
                'z_position': self._intersection_levels[i],
                'exposure_time': exposure_time,
                'light_off_time': self._resin.exposure_settings.light_off_delay,
            }
            layer_data.append(layer_info)
        
        return layer_data

    
    def _encode_layer_rle(self, image: np.ndarray) -> tuple[bytes, int]:
        """
        Encode a layer image using RLE compression as specified in the Goo format
        
        The encoding works as follows:
        - Each run is encoded as: run_length (variable bytes) + grayscale_value (1 byte)
        - Run lengths use a variable-length scheme with 7-bit chunks:
          - Bytes with high bit set (0x80) indicate continuation
          - Last byte has high bit clear

        Parameters
        ----------
        image : np.ndarray
            Grayscale image as 2D numpy array
            
        Returns
        -------
        tuple[bytes, int]
            Encoded data and checksum (wrapping sum, then negated as u8)
        """
        # Flatten the image to 1D array and ensure uint8
        if len(image.shape) == 2:
            flat_image = image.flatten().astype(np.uint8)
        else:
            flat_image = image.astype(np.uint8)
            
        if len(flat_image) == 0:
            return bytes(), 0
        
        # Use NumPy for efficient RLE encoding
        change_indices = np.where(np.diff(flat_image) != 0)[0] + 1
        
        # Split array at change points to get runs
        run_starts = np.concatenate(([0], change_indices))
        run_ends = np.concatenate((change_indices, [len(flat_image)]))
        run_lengths = run_ends - run_starts
        run_values = flat_image[run_starts]
        
        # Encode runs
        encoded = bytearray()
        for length, value in zip(run_lengths, run_values):
            encoded.extend(self._encode_run_length(int(length)))
            encoded.append(int(value))
        
        # Calculate checksum: wrapping sum of all bytes, then negated
        checksum_sum = sum(encoded) & 0xFF
        checksum = (~checksum_sum) & 0xFF
        
        return bytes(encoded), checksum
    

    def _encode_run_length(self, length: int) -> bytes:
        """
        Encode run length with variable-length encoding (7-bit chunks)
        
        The encoding uses 7 bits per byte with the high bit as a continuation flag:
        - If high bit is set (0x80), more bytes follow
        - If high bit is clear, this is the last byte
        
        Parameters
        ----------
        length : int
            Run length to encode
            
        Returns
        -------
        bytes
            Encoded run length
        """
        if length <= 0x7F:
            # Single byte: length with high bit clear
            return bytes([length])
        
        # Multi-byte encoding: 7 bits per byte, high bit set on all but last
        result = []
        remaining = length
        
        while remaining > 0x7F:
            # Take lower 7 bits, set high bit for continuation
            result.append((remaining & 0x7F) | 0x80)
            remaining >>= 7
        
        # Last byte: no continuation bit
        result.append(remaining & 0x7F)
        
        return bytes(result)
    

    def _create_preview_images(self) -> tuple[bytes, bytes]:
        """
        Create preview images with text info embedded in the header.
        
        The preview contains:
        - Filename
        - Layer count
        - Total print time
        
        Returns two preview images in RGB565 format (little-endian):
        - Small preview: 116x116 pixels (26912 bytes)
        - Large preview: 290x290 pixels (168200 bytes)
        
        Returns
        -------
        tuple[bytes, bytes]
            Small preview and large preview as raw RGB565 data
        """

        preview_sizes = [(116, 116), (290, 290)]
        previews = []
        
        # Calculate total print time
        total_print_time_s = sum(
            layer['exposure_time'] + layer['light_off_time'] for layer in self._layer_data
        )
        
        # Format print time
        hours, remainder = divmod(total_print_time_s, 3600)
        minutes, _ = divmod(remainder, 60)
        time_str = f"{int(hours):02d}h {int(minutes):02d}m"

        # Prepare text lines
        line1 = self.file_name
        line2 = f"Layers: {len(self._layers)}"
        line3 = f"Time: {time_str}"
        
        for width, height in preview_sizes:
            # Create a new black image
            preview = Image.new('RGB', (width, height), 'black')
            draw = ImageDraw.Draw(preview)
            
            # Try to load a default font, fallback to a basic one
            try:
                # Adjust font size based on image width
                font_size = max(10, int(width / 12))
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            # Calculate text positions
            y_pos = height * 0.25
            line_height = font.getbbox("A")[3] * 1.2

            # Draw text on the image
            draw.text((width / 2, y_pos), line1, font=font, fill="white", anchor="ms")
            y_pos += line_height
            draw.text((width / 2, y_pos), line2, font=font, fill="white", anchor="ms")
            y_pos += line_height
            draw.text((width / 2, y_pos), line3, font=font, fill="white", anchor="ms")
            
            # Encode as RGB565 format
            preview_data = self._encode_preview_rgb565(preview)
            previews.append(preview_data)
        
        return previews[0], previews[1]
    
    

    def _encode_preview_rgb565(self, image: Image.Image) -> bytes:
        """
        Encode preview image to RGB565 format (16-bit color, big-endian)
        
        Parameters
        ----------
        image : Image.Image
            RGB image to encode
            
        Returns
        -------
        bytes
            Encoded image data
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        encoded = bytearray()
        
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x, :3]
                
                # Convert 8-bit RGB to 5-6-5 format
                r5 = (r >> 3) & 0x1F
                g6 = (g >> 2) & 0x3F
                b5 = (b >> 3) & 0x1F
                
                # Pack into 16-bit value
                rgb565 = (r5 << 11) | (g6 << 5) | b5
                
                # Write as little-endian 16-bit value
                encoded.append(rgb565 & 0xFF)
                encoded.append((rgb565 >> 8) & 0xFF)
        
        return bytes(encoded)
    

    def _write_sized_string(self, s: str, size: int) -> bytes:
        """
        Write a fixed-size string (null-padded)
        
        Parameters
        ----------
        s : str
            String to write
        size : int
            Fixed size in bytes
            
        Returns
        -------
        bytes
            Null-padded string
        """
        s_bytes = s.encode('ascii', errors='replace')[:size]
        return s_bytes.ljust(size, b'\x00')
    

    def _build_header(self) -> bytes:
        """
        Build the Goo file header according to the format specification
        
        Header structure (based on Rust HeaderInfo struct):
        All values are BIG-ENDIAN (not mentioned in spec but critical!)
        
        Returns
        -------
        bytes
            Complete header data
        """
        header = bytearray()
        
        # Get printer settings
        res_x, res_y = self._printer.settings.screen_resolution
        # Calculate build volume from screen resolution and pixel size
        build_x = res_x * self._printer.settings.pixel_size
        build_y = res_y * self._printer.settings.pixel_size
        build_z = self._printer.settings.max_z_height
        
        # Calculate print statistics
        total_layers = len(self._layers)
        
        # Calculate total print time in seconds (assuming the movement times are included in the light off time)
        total_print_time = 0.0
        for i, layer_data in enumerate(self._layer_data):
            total_print_time += layer_data['exposure_time']
            total_print_time += layer_data['light_off_time']
        
        # Version (4 bytes string) - from hex dump: "V3.0"
        header.extend(self._write_sized_string("V3.0", 4))
        
        # MAGIC_TAG (8 bytes) - CRITICAL: comes after version!
        header.extend(self.MAGIC_TAG)
        
        # Software info (32 bytes) - from hex dump shows "UBox"
        header.extend(self._write_sized_string("OpenSLAice", 32))
        
        # Software version (24 bytes) - from hex dump shows "v2.3"
        header.extend(self._write_sized_string("v1.0", 24))
        
        # File time (24 bytes) - formatted datetime
        file_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header.extend(self._write_sized_string(file_time, 24))
        
        # Printer name (32 bytes) - from hex dump shows printer name
        header.extend(self._write_sized_string(self._printer.name, 32))
        
        # Printer type (32 bytes) - from hex dump shows "DLP"
        header.extend(self._write_sized_string("DLP", 32))
        
        # Profile name (32 bytes) - from hex dump shows "Profile"
        header.extend(self._write_sized_string("Profile", 32))
        
        # Anti-aliasing level (2 bytes, u16)
        aa_level = self._resin.special_settings.anti_aliasing
        header.extend(struct.pack('<H', aa_level))
        
        # Grey level (2 bytes, u16)
        header.extend(struct.pack('<H', 0)) # Not used
        
        # Blur level (2 bytes, u16)
        header.extend(struct.pack('<H', 0)) # Not used
        
        # Small preview image (116x116 RGB565 = 26912 bytes)
        small_preview, large_preview = self._create_preview_images()
        header.extend(small_preview)
        
        # DELIMITER after small preview
        header.extend(self.DELIMITER)
        
        # Large preview image (290x290 RGB565 = 168200 bytes)
        header.extend(large_preview)
        
        # DELIMITER after large preview
        header.extend(self.DELIMITER)
        
        # Layer count (4 bytes, u32)
        header.extend(struct.pack('<I', total_layers))
        
        # X resolution (2 bytes, u16)
        header.extend(struct.pack('<H', res_x))
        
        # Y resolution (2 bytes, u16)
        header.extend(struct.pack('<H', res_y))
        
        # X mirror (1 byte, bool)
        header.extend(struct.pack('<?', False)) # Not used

        # Y mirror (1 byte, bool)
        header.extend(struct.pack('<?', False)) # Not used
        
        # X size in mm (4 bytes, f32)
        header.extend(struct.pack('<f', build_x))
        
        # Y size in mm (4 bytes, f32)
        header.extend(struct.pack('<f', build_y))
        
        # Z size in mm (4 bytes, f32)
        header.extend(struct.pack('<f', build_z))
        
        # Layer thickness (4 bytes, f32) - use first layer's thickness as assumption
        layer_thickness = self._layer_heights[0] if len(self._layer_heights) > 0 else 0.05
        header.extend(struct.pack('<f', layer_thickness))
        
        # Exposure time (4 bytes, f32)
        normal_exposure = self._resin.get_exp_time(self._resin.exposure_settings.bottom_layers, layer_thickness)
        header.extend(struct.pack('<f', normal_exposure))
        
        # Exposure delay mode (1 byte, bool)
        header.extend(struct.pack('<?', False)) # Not used
        
        # Turn off time / light-off delay (4 bytes, f32)
        header.extend(struct.pack('<f', self._resin.exposure_settings.light_off_delay))
        
        # Bottom before lift time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used
        
        # Bottom after lift time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used
        
        # Bottom after retract time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used

        # Before lift time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used
        
        # After lift time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used

        # After retract time (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # Not used
        
        # Bottom exposure time (4 bytes, f32) from the first layer as assumption
        bottom_exposure = self._resin.get_exp_time(0, layer_thickness)
        header.extend(struct.pack('<f', bottom_exposure))
        
        # Bottom layers (4 bytes, u32)
        header.extend(struct.pack('<I', self._resin.exposure_settings.bottom_layers))
        
        # Bottom lift distance (4 bytes, f32)
        header.extend(struct.pack('<f', self._resin.movement_settings.bottom_lift_height))
        
        # Bottom lift speed (4 bytes, f32) - mm/min
        header.extend(struct.pack('<f', self._resin.movement_settings.bottom_lift_speed))
        
        # Lift distance (4 bytes, f32)
        header.extend(struct.pack('<f', self._resin.movement_settings.normal_lift_height))
        
        # Lift speed (4 bytes, f32) - mm/min
        header.extend(struct.pack('<f', self._resin.movement_settings.normal_lift_speed))
        
        # Bottom retract distance (4 bytes, f32)
        header.extend(struct.pack('<f', self._resin.movement_settings.bottom_lift_height))
        
        # Bottom retract speed (4 bytes, f32) - mm/min
        header.extend(struct.pack('<f', self._resin.movement_settings.bottom_retract_speed))
        
        # Retract distance (4 bytes, f32)
        header.extend(struct.pack('<f', self._resin.movement_settings.normal_lift_height))
        
        # Retract speed (4 bytes, f32) - mm/min
        header.extend(struct.pack('<f', self._resin.movement_settings.normal_retract_speed))
        
        # Bottom second lift distance (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Bottom second lift speed (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Second lift distance (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Second lift speed (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Bottom second retract distance (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Bottom second retract speed (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used

        # Second retract distance (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used

        # Second retract speed (4 bytes, f32)
        header.extend(struct.pack('<f', 0.0)) # currently not used
        
        # Bottom light PWM (2 bytes, u16)
        header.extend(struct.pack('<H', 255)) # Max brightness
        
        # Light PWM (2 bytes, u16)
        header.extend(struct.pack('<H', 255)) # Max brightness
        
        # Advance mode (1 byte, bool) - enables individual layer settings for individual exposure times
        header.extend(struct.pack('<?', True))
        
        # Printing time in seconds (4 bytes, u32)
        header.extend(struct.pack('<I', int(total_print_time)))
        
        # Total volume in ml (4 bytes, f32)
        header.extend(struct.pack('<f', self._volume / 1000.0))
        
        # Total weight in grams (4 bytes, f32)
        total_weight = (self._volume / 1000.0) * self._resin.properties.density
        header.extend(struct.pack('<f', total_weight))
        
        # Total price (4 bytes, f32)
        total_price = (self._volume / 1000.0) * self._resin.properties.price
        header.extend(struct.pack('<f', total_price))
        
        # Price unit (8 bytes string)
        header.extend(self._write_sized_string(self._resin.properties.currency, 8))
        
        # Header size (4 bytes, u32) - CRITICAL: must write header size value here!
        header.extend(struct.pack('<I', self.HEADER_SIZE))
        
        # Grey scale level (1 byte, bool)
        header.extend(struct.pack('<?', False)) # Not used
        
        # Transition layers (2 bytes, u16)
        header.extend(struct.pack('<H', self._resin.exposure_settings.transition_layers)) 
        
        # Pad to HEADER_SIZE (195477 bytes)
        if len(header) < self.HEADER_SIZE:
            header.extend(b'\x00' * (self.HEADER_SIZE - len(header)))
        elif len(header) > self.HEADER_SIZE:
            # Truncate if somehow too long
            header = header[:self.HEADER_SIZE]
        
        return bytes(header)
    

    def _build_layer_definition(self, layer_idx: int) -> bytes:
        """
        Build layer definition structure matching Rust LayerContent::serialize
        
        Layer format (from Rust):
        - pause_flag (2 bytes, u16)
        - pause_position_z (4 bytes, f32)
        - layer_position_z (4 bytes, f32)
        - layer_exposure_time (4 bytes, f32)
        - layer_off_time (4 bytes, f32)
        - before_lift_time, after_lift_time, after_retract_time (f32 each)
        - lift_distance, lift_speed (f32 each)
        - second_lift_distance, second_lift_speed (f32 each)
        - retract_distance, retract_speed (f32 each)
        - second_retract_distance, second_retract_speed (f32 each)
        - light_pwm (2 bytes, u16)
        - DELIMITER
        - data_len (4 bytes, u32) = len(data) + 2 (includes 0x55 and checksum)
        - 0x55 magic byte
        - image data
        - checksum (1 byte, u8)
        - DELIMITER
        
        Parameters
        ----------
        layer_idx : int
            Index of the layer
            
        Returns
        -------
        bytes
            Complete layer data
        """
        layer_info = self._layer_data[layer_idx]
        layer_def = bytearray()
        
        # Pause flag (u16) - not used
        layer_def.extend(struct.pack('<H', 0))
        
        # Pause position Z (f32) - not used
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Layer position Z (f32)
        layer_def.extend(struct.pack('<f', layer_info['z_position']))
        
        # Layer exposure time (f32)
        layer_def.extend(struct.pack('<f', layer_info['exposure_time']))
        
        # Layer off time (f32)
        layer_def.extend(struct.pack('<f', layer_info['light_off_time']))
        
        # Before lift time (f32) - using defaults
        layer_def.extend(struct.pack('<f', 0.0))
        
        # After lift time (f32)
        layer_def.extend(struct.pack('<f', 0.0))
        
        # After retract time (f32)
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Lift distance (f32)
        is_bottom = layer_idx < self._resin.exposure_settings.bottom_layers
        lift_dist = self._resin.movement_settings.bottom_lift_height if is_bottom else self._resin.movement_settings.normal_lift_height
        layer_def.extend(struct.pack('<f', lift_dist))
        
        # Lift speed (f32)
        lift_speed = self._resin.movement_settings.bottom_lift_speed if is_bottom else self._resin.movement_settings.normal_lift_speed
        layer_def.extend(struct.pack('<f', lift_speed))
        
        # Second lift distance (f32) - not used
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Second lift speed (f32) - not used
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Retract distance (f32)
        layer_def.extend(struct.pack('<f', lift_dist))
        
        # Retract speed (f32)
        retract_speed = self._resin.movement_settings.bottom_retract_speed if is_bottom else self._resin.movement_settings.normal_retract_speed
        layer_def.extend(struct.pack('<f', retract_speed))
        
        # Second retract distance (f32) - not used
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Second retract speed (f32) - not used
        layer_def.extend(struct.pack('<f', 0.0))
        
        # Light PWM (u16)
        layer_def.extend(struct.pack('<H', self._resin.special_settings.light_pwm))
        
        # DELIMITER
        layer_def.extend(self.DELIMITER)
        
        # Data length (u32) = len(data) + 2 (for 0x55 and checksum)
        data_len = len(layer_info['data']) + 2
        layer_def.extend(struct.pack('<I', data_len))
        
        # Magic byte 0x55
        layer_def.append(0x55)
        
        # Image data
        layer_def.extend(layer_info['data'])
        
        # Checksum (u8) - already calculated correctly in _encode_layer_rle
        layer_def.append(layer_info['checksum'])
        
        # DELIMITER
        layer_def.extend(self.DELIMITER)
        
        return bytes(layer_def)
    

    def save(self, save_path: str) -> bool:
        """
        Save the Goo file to specified path
        
        File structure:
        - Header (0x2FB95 = 195477 bytes, includes preview images)
        - Layer definitions (24 bytes per layer, with individual exposure settings)
        - Layer data (RLE encoded, variable size)
        
        Parameters
        ----------
        save_path : str
            Directory path where the file should be saved
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        try:
            # Build file path
            if not save_path.endswith('.goo'):
                file_path = os.path.join(save_path, f"{self.file_name}.goo")
            else:
                file_path = save_path
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            
            # Build header
            header = self._build_header()
            
            # Write file
            with open(file_path, 'wb') as f:
                # Write header (includes preview images)
                f.write(header)
                
                # Write layers (each layer definition includes data, delimiters, etc.)
                for i in range(len(self._layers)):
                    layer_def = self._build_layer_definition(i)
                    f.write(layer_def)
                
                # Write ending string at the end of the file
                f.write(self.ENDING_STRING)
            
            print(f"Successfully saved Goo file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving Goo file: {e}")
            import traceback
            traceback.print_exc()
            return False


# Register the GooFile class in the FileTypeRegistry
FileTypeRegistry.register(GooFile)
