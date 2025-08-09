# Tests Directory

This directory contains test scripts for the OpenBadge Creator & Lookup Service.

## Available Tests

### 1. Style Templates Test (`test_style_templates.py`)
Tests the badge style templates functionality, including:
- Verification that style templates were successfully moved to `constants.py`
- Testing of all 6 available badge styles: professional, modern, artistic, classic, superhero, retro
- Integration testing with the MCP client

**Run the test:**
```bash
python tests/test_style_templates.py
```

### 2. Example Signed Badge (`example_signed_badge.py`)
Demonstrates and tests the cryptographic signing functionality:
- Creates example signed Open Badge 3.0 credentials
- Shows verification method implementations
- Demonstrates cryptographic proof generation

**Run the example:**
```bash
python tests/example_signed_badge.py
```

### 3. Dual Image System Test (`test_dual_image_system.py`)
Tests the dual-image badge system functionality:
- Verifies separation of 512x512 badge images and credential certificates
- Tests API endpoints for both image types
- Validates updated badge creation and lookup functions
- Confirms enhanced Gradio interface with dual upload controls

**Run the test:**
```bash
python tests/test_dual_image_system.py
```

### 4. Dual Image System Demo (`demo_dual_image_system.py`)
Comprehensive demonstration of the dual-image badge system:
- Shows complete workflow from creation to lookup
- Demonstrates all 6 badge styles including superhero and retro
- Verifies API endpoints and storage structure
- Explains the complete feature set

**Run the demo:**
```bash
python tests/demo_dual_image_system.py
```

### 5. App Style Integration Test (`test_app_style_integration.py`)
Tests the integration between app.py and the STYLE_TEMPLATES from constants.py:
- Verifies that app.py correctly imports and uses STYLE_TEMPLATES
- Tests dynamic dropdown choices in Gradio interface
- Validates that all 6 styles are functional
- Confirms dynamic documentation updates

**Run the test:**
```bash
python tests/test_app_style_integration.py
```

### 6. Byte String Conversion Test (`test_byte_string_fix.py`)
Tests the byte string to PIL image conversion functionality for MCP servers:
- Verifies base64 string to PIL image conversion
- Tests raw byte string to PIL image conversion
- Validates MCP result extraction with user's exact response format
- Tests URL text handling from MCP server responses
- Confirms URL-only text parsing and download logic
- Ensures Union[str, Image.Image] return type consistency
- Validates support for WebP, PNG, JPEG formats
- Tests MCP server gradio_api/file URL format handling

**Run the test:**
```bash
python tests/test_byte_string_fix.py
```

## Running All Tests

To run all tests from the project root directory:

```bash
# Run style templates test
python tests/test_style_templates.py

# Run signed badge example
python tests/example_signed_badge.py

# Run dual image system test  
python tests/test_dual_image_system.py

# Run dual image system demo
python tests/demo_dual_image_system.py

# Run app style integration test
python tests/test_app_style_integration.py

# Run byte string conversion test
python tests/test_byte_string_fix.py
```

## Test Structure

- Tests are located in the `tests/` folder
- Each test file is self-contained and can be run independently
- Tests include proper error handling and detailed output
- Import paths are configured to work from the tests directory

## Adding New Tests

When adding new test files:
1. Place them in the `tests/` directory
2. Add proper path handling for module imports:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
3. Follow the existing naming convention (`test_*.py`)
4. Include descriptive output and proper error handling
5. Update this README with information about the new test