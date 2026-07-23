"""
Script to generate a small synthetic YUV420 test video for DCVC-RT testing.
Creates a 416x240 video with 10 frames containing moving gradient patterns.
"""
import numpy as np
import os

# Video parameters
width = 416
height = 240
num_frames = 10

# Output path
output_dir = os.path.join("e:/LAB/DCVC RT/data/TEST_SMALL")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_416x240.yuv")

# YUV420 format: Y plane is full resolution, U and V are half resolution
y_size = width * height
uv_width = width // 2
uv_height = height // 2
uv_size = uv_width * uv_height

print(f"Generating synthetic YUV420 video: {width}x{height}, {num_frames} frames")
print(f"Output: {output_path}")
print(f"Y plane: {width}x{height} = {y_size} bytes per frame")
print(f"U/V planes: {uv_width}x{uv_height} = {uv_size} bytes each per frame")
print(f"Total per frame: {y_size + 2 * uv_size} bytes")

with open(output_path, 'wb') as f:
    for i in range(num_frames):
        # Generate Y (luminance) plane - moving diagonal gradient
        y_plane = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            for col in range(width):
                # Create a moving pattern: diagonal gradient + circular motion
                val = int((col + row + i * 20) % 256)
                # Add some texture variation
                val = int(val * 0.7 + 40 * np.sin(col / 30.0 + i * 0.5) + 40 * np.cos(row / 25.0 + i * 0.3))
                y_plane[row, col] = np.clip(val, 16, 235)  # TV range

        # Generate U (Cb) plane - half resolution, slight color shift
        u_plane = np.zeros((uv_height, uv_width), dtype=np.uint8)
        for row in range(uv_height):
            for col in range(uv_width):
                val = 128 + int(30 * np.sin((col + i * 10) / 20.0) * np.cos((row + i * 5) / 15.0))
                u_plane[row, col] = np.clip(val, 16, 240)

        # Generate V (Cr) plane - half resolution, different color shift
        v_plane = np.zeros((uv_height, uv_width), dtype=np.uint8)
        for row in range(uv_height):
            for col in range(uv_width):
                val = 128 + int(25 * np.cos((col + i * 8) / 25.0) * np.sin((row + i * 6) / 20.0))
                v_plane[row, col] = np.clip(val, 16, 240)

        # Write in YUV420 planar format: Y, then U, then V
        f.write(y_plane.tobytes())
        f.write(u_plane.tobytes())
        f.write(v_plane.tobytes())

        print(f"  Frame {i+1}/{num_frames} generated")

file_size = os.path.getsize(output_path)
expected_size = num_frames * (y_size + 2 * uv_size)
print(f"\nDone! File size: {file_size} bytes (expected: {expected_size})")
print(f"File: {output_path}")
