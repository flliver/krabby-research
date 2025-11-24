# Wyatt House Asset

This directory contains the Wyatt House asset files - a simple house model created for use in the parkour environment.

## Files

### Reference Files
- **wyattHouse.usda** - Hand-authored reference file containing the Wyatt House design. This is a simple house with:
  - Four walls (5m x 4m footprint)
  - Pyramidal roof (3m tall walls, 4.5m peak)
  - Front door (1.2m wide, 2.2m tall)
  - Two side windows (1m x 1m)
  - Materials for walls, roof, door, and windows

### Generated Files
- **wyattHouse_generated.usda** - Programmatically generated version of the Wyatt House
  - Generated from the same specifications as the reference file
  - Validates the generation pipeline
  - Can be used interchangeably with the reference file

### Generator Script
- **../../scripts/gen_wyatt_house.py** - Python script to generate the Wyatt House asset

## Usage

### Viewing the Asset
The USDA files can be opened in any USD-compatible viewer such as:
- NVIDIA Omniverse
- USD View (from Pixar's USD distribution)
- Isaac Sim

### Generating the Asset Programmatically

```bash
cd parkour/scripts
python3 gen_wyatt_house.py --output ../assets/scenes/wyattHouse_generated.usda --overwrite
```

#### Command-line Options
- `--output PATH` - Specify the output file path (default: ../assets/scenes/wyattHouse_generated.usda)
- `--overwrite` - Allow overwriting existing files

### Requirements for Generation Script
The generation script requires the USD Python bindings (pxr module):
- NVIDIA Omniverse with USD
- Isaac Sim Python environment
- Standalone USD Python bindings from Pixar

## Asset Specifications

### Dimensions
- **Footprint**: 5m (length) × 4m (width)
- **Wall Height**: 3m
- **Total Height**: 4.5m (including roof peak)
- **Door**: 1.2m wide × 2.2m tall
- **Windows**: 1m wide × 1m tall (each)

### Coordinate System
- **Up Axis**: Z-axis
- **Units**: Meters
- **Origin**: Centered at ground level (Z=0)

### Materials
1. **WallMaterial**: Light gray walls (RGB: 0.85, 0.85, 0.90)
2. **RoofMaterial**: Brown roof (RGB: 0.4, 0.2, 0.15)
3. **DoorMaterial**: Brown door (RGB: 0.5, 0.3, 0.1)
4. **WindowMaterial**: Semi-transparent blue windows (RGB: 0.2, 0.3, 0.5, Opacity: 0.4)

## Design Philosophy

The Wyatt House is designed to be:
- **Simple**: Basic geometric shapes for easy rendering and collision detection
- **Lightweight**: Minimal polygon count for efficient simulation
- **Realistic Scale**: Appropriate dimensions for robot navigation and parkour tasks
- **Modular**: Easy to modify or extend with additional features

## Integration with Parkour Environment

This asset can be used in the parkour environment to:
- Provide obstacles for navigation
- Test robot perception systems
- Create realistic urban/suburban scenarios
- Serve as a landmark for waypoint navigation

## Comparison with Other House Assets

| Asset | Complexity | Features | Use Case |
|-------|-----------|----------|----------|
| house_model_reference.usda | High | Detailed environment with trees, fence, walkway | Full scene simulation |
| wyattHouse.usda | Low | Basic house structure | Simple obstacle/landmark |

## Future Enhancements

Potential additions to the Wyatt House asset:
- [ ] Floor/ground plane
- [ ] Interior geometry
- [ ] Driveway or walkway
- [ ] Landscaping elements
- [ ] Multiple house variants
- [ ] Procedural generation options
