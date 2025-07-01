# 🏆 OpenBadge Creator & Lookup Service

A Hugging Face Space for creating and looking up Open Badge 3.0 compliant digital credentials.

## Features

- **Create Open Badges**: Generate Open Badge 3.0 compliant digital credentials
- **Badge Lookup**: Look up badges by GUID via web interface or API
- **Metadata Embedding**: Embed badge metadata directly into PNG images
- **REST API**: Programmatic access to badge data
- **Persistent Storage**: Store badges in Hugging Face datasets

## Usage

### Web Interface

1. **Create Badge**: Use the "Create Badge" tab to generate new digital credentials
2. **Look Up Badge**: Use the "Look Up Badge" tab to find existing badges by GUID

### API Access

Access badges programmatically using the REST API:

```python
import requests

# Get complete badge information
response = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}")
badge_data = response.json()

# Get only metadata
metadata = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/metadata").json()

# Get badge image URL
image_response = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/image")
```

### URL Structure

- **Badge Lookup**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}`
- **Metadata Only**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/metadata`
- **Badge Image**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/image`

## Open Badge 3.0 Compliance

This service creates badges that comply with the [Open Badge 3.0 specification](https://www.imsglobal.org/spec/ob/v3p0/):

- JSON-LD format with proper context
- Verifiable Credentials structure
- Achievement and AchievementSubject definitions
- Embedded metadata in PNG images using iTXt chunks

## Storage

Badges are stored in the Hugging Face dataset repository with the following structure:

```
badges/
├── {guid-1}/
│   ├── user.json          # Badge metadata
│   └── badge.png          # Badge image with embedded metadata
├── {guid-2}/
│   ├── user.json
│   └── badge.png
└── ...
```

## Environment Setup

Required environment variables:

- `HF_TOKEN`: Hugging Face API token for repository access
- `HF_REPO_ID`: Target repository ID (default: "Surn/Storage")

## Technical Details

### Dependencies

- **Gradio 5.35.0**: Web interface and API framework
- **FastAPI**: REST API endpoints
- **Hugging Face Hub**: Repository storage and access
- **PIL (Pillow)**: Image processing and metadata embedding

### Modules

- `modules/build_openbadge_metadata.py`: Open Badge 3.0 metadata generation
- `modules/add_openbadge_metadata.py`: PNG metadata embedding
- `modules/storage.py`: Hugging Face repository operations
- `modules/constants.py`: Configuration and constants

## License

This project is open source and available under standard licensing terms.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

Built with ❤️ for the digital credentialing community.