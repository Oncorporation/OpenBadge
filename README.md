---
title: OpenBadge Creator & Lookup Service
emoji: 🏆
short_description: Creating and looking up Open Badge 3.0 compliant digital credentials with cryptographic verification
colorFrom: yellow
colorTo: red
python_version: 3.10.13
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
license: apache-2.0
hf_oauth: false
fullWidth: true
tags:
- Open Badge 3.0
- digital credentials
- cryptographic verification
- verifiable credentials
- Ed25519
- PyNaCl
- storage
- API
- MCP
- Hugging Face
- gradio
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6346595c9e5f0fe83fc60444/cyAmkkrQsfDjolwuHb4ZX.png
---

# 🏆 OpenBadge Creator & Lookup Service

A comprehensive Hugging Face Space for creating, signing, and verifying Open Badge 3.0 compliant digital credentials with cryptographic proof capabilities.

## ✨ Features

- **🔐 Cryptographic Verification**: Full Ed25519 digital signature support with verification methods
- **🔑 Secure Key Management**: Encrypted private key storage in private Hugging Face repositories
- **📜 Create Open Badges**: Generate Open Badge 3.0 compliant digital credentials
- **🔍 Badge Lookup**: Look up badges by GUID via web interface or REST API
- **📊 Metadata Embedding**: Embed badge metadata directly into PNG images using iTXt chunks
- **🌐 REST API**: Comprehensive programmatic access to badge data and verification
- **💾 Persistent Storage**: Secure storage in Hugging Face datasets with organized structure
- **✅ Standards Compliance**: Full adherence to Open Badge 3.0 and W3C Verifiable Credentials specifications

## 🚀 Usage

### Web Interface

1. **Create Badge**: Use the "Create Badge" tab to generate new cryptographically signed digital credentials
2. **Look Up Badge**: Use the "Look Up Badge" tab to find and verify existing badges by GUID
3. **API Documentation**: Built-in documentation for programmatic access

### REST API Access

Access badges programmatically using the comprehensive REST API:
import requests

# Get complete badge information (includes verification method and proof)
response = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}")
badge_data = response.json()

# Get only metadata (includes cryptographic proof)
metadata = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/metadata").json()

# Get badge image (with embedded metadata)
image_response = requests.get("https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/image")

# Verify the cryptographic signature
verification_method = badge_data['metadata']['verificationMethod'][0]
proof = badge_data['metadata']['proof']
### API Endpoints

- **Badge Lookup**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}`
- **Metadata Only**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/metadata`
- **Badge Image**: `https://huggingface.co/spaces/surn/OpenBadge/badge/{guid}/image`

## 🔒 Open Badge 3.0 & Cryptographic Compliance

This service creates badges that fully comply with the [Open Badge 3.0 specification](https://www.imsglobal.org/spec/ob/v3p0/) and [W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/):

### Core Features
- **JSON-LD format** with proper `@context` declarations
- **Verifiable Credentials structure** with full credential lifecycle support
- **Achievement and AchievementSubject** definitions per specification
- **Embedded metadata** in PNG images using iTXt chunks for offline verification

### Cryptographic Features
- **Ed25519 Digital Signatures** using PyNaCl (libsodium) for maximum security
- **Verification Methods** with multibase-encoded public keys
- **Cryptographic Proofs** using `Ed25519Signature2020` proof type
- **Data Integrity** with SHA-256 hashing and canonical JSON serialization
- **Secure Key Management** with encrypted private key storage

### Supported Verification Methods
- **Ed25519VerificationKey2020** (recommended): High-performance elliptic curve signatures
- **JsonWebKey2020**: Support for RSA and secp256k1 algorithms
- **Multibase encoding**: Base58btc encoding for keys and signatures

## 📁 Storage Structure

Badges and cryptographic keys are stored in a **PRIVATE** Hugging Face dataset repository with a secure, organized structure:

⚠️ **WARNING**: The repository MUST be private to protect encrypted private keys.

badges/
├── {guid-1}/
│   ├── user.json          # Complete badge metadata with cryptographic proof
│   └── badge.png          # Badge image with embedded metadata
├── {guid-2}/
│   ├── user.json          # Includes verification method and signature
│   └── badge.png          # Self-contained verifiable credential
└── ...

keys/
├── issuers/
│   ├── {issuer-id}/
│   │   ├── private_key.json    # Encrypted private key
│   │   └── public_key.json     # Public key for verification
│   └── {issuer-id-2}/
│       ├── private_key.json
│       └── public_key.json
└── global/
    └── verification_methods.json  # Registry of all verification methods

### Key Storage Details
- **Private keys**: Encrypted using basic XOR encryption (demo) - **use Fernet encryption in production**
- **Public keys**: Stored in plain text for verification purposes
- **Registry**: Global verification methods registry for efficient key discovery
- **Issuer folders**: Sanitized issuer IDs as folder names for file system compatibility

## ⚙️ Environment Setup

### Required Environment Variables
- `HF_TOKEN`: Hugging Face API token for repository access
- `HF_REPO_ID`: Target repository ID (default: "Surn/Storage")

### 🔐 **CRITICAL SECURITY REQUIREMENT**
**The target repository (`HF_REPO_ID`) MUST be a PRIVATE Hugging Face repository** to ensure the security of stored encrypted private keys. **Never use this service with public repositories** as it would expose cryptographic key material.

### Optional Configuration
- `TMPDIR`: Temporary directory for file processing
- `SPACE_NAME`: Hugging Face Space identifier

## 🛠️ Technical Details

### Core Dependencies
- **Gradio 5.37.0**: Modern web interface with MCP server support
- **FastAPI**: High-performance REST API framework
- **PyNaCl**: Cryptographic library for Ed25519 signatures
- **Multibase**: Encoding for cryptographic keys and signatures
- **Hugging Face Hub**: Repository storage and access
- **PIL (Pillow)**: Image processing and metadata embedding

### Architecture Modules
- **`modules/build_openbadge_metadata.py`**: Open Badge 3.0 metadata generation with cryptographic support
- **`modules/crypto_utils.py`**: Ed25519 key generation, signing, and verification utilities
- **`modules/storage.py`**: Hugging Face repository operations, key management, and file storage
- **`modules/add_openbadge_metadata.py`**: PNG metadata embedding with iTXt chunks
- **`modules/constants.py`**: Configuration constants and environment variables
- **`modules/file_utils.py`**: File handling and download utilities

### Key Management Functions
- **`store_issuer_keypair()`**: Securely store issuer cryptographic keys
- **`get_issuer_keypair()`**: Retrieve and decrypt issuer keys
- **`derive_public_key_from_private()`**: Derive public key from private Ed25519 key
- **`get_verification_methods_registry()`**: Access global verification methods
- **`list_issuer_ids()`**: List all registered issuers

### Security Features
- **Secure key generation** using cryptographically secure random number generators
- **Encrypted private key storage** with access-controlled private repositories
- **Key derivation** from issuer identifiers with sanitization
- **Verification method registry** for efficient key discovery
- **Signature verification** capabilities for credential validation
- **Canonical JSON** serialization for consistent hashing

## 🚧 Implementation Status

### ✅ Completed Features
- Ed25519 key generation and signing
- Private key derivation from public keys
- Encrypted key storage in private repositories
- Open Badge 3.0 metadata generation
- PNG metadata embedding
- REST API endpoints
- Badge lookup and verification

### 🔄 In Development
- Integration of cryptographic signing in main badge creation workflow
- Enhanced key rotation mechanisms
- Production-grade Fernet encryption

### 📋 Planned Features
- Multiple signature algorithm support
- Key revocation mechanisms
- Batch badge operations
- Enhanced verification APIs

## 🔧 Development

### Installing Dependencies
````````
pip install -r requirements.txt
````````
### Key Dependencies Added
- `pynacl`: Ed25519 cryptographic operations
- `multibase`: Key and signature encoding
- `python-dotenv`: Environment variable management

### Running Locally
````````
python app.py
````````
The application will launch with:
- **Web Interface**: Interactive badge creation and lookup
- **REST API**: Programmatic access endpoints
- **MCP Server**: Model Context Protocol support for AI integration

### 🚨 Security Considerations

1. **Private Repository**: Always use private repositories for key storage
2. **Key Encryption**: Current implementation uses basic XOR encryption (demo) - **upgrade to Fernet encryption in production**
3. **Access Control**: Implement proper authentication and authorization
4. **Key Rotation**: Regularly rotate cryptographic keys
5. **Audit Logging**: Monitor key access and usage patterns
6. **Repository Privacy**: Ensure `HF_REPO_ID` points to a private repository
7. **Token Security**: Protect your `HF_TOKEN` with appropriate access controls

## 📚 Standards Compliance

This implementation follows current best practices:

- **[Open Badge 3.0](https://www.imsglobal.org/spec/ob/v3p0/)**: Full specification compliance
- **[W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/)**: Data model and proof formats
- **[W3C DID Core](https://www.w3.org/TR/did-core/)**: Decentralized identifier support
- **[Data Integrity](https://www.w3.org/TR/vc-data-integrity/)**: Cryptographic proof mechanisms
- **[Multibase](https://datatracker.ietf.org/doc/html/draft-multiformats-multibase)**: Key encoding standards

## 🤝 Contributing

We welcome contributions to enhance the OpenBadge Creator & Lookup Service!

### Areas for Contribution
- Enhanced encryption methods (Fernet implementation)
- Additional cryptographic algorithm support
- Key rotation and management features
- Enhanced verification methods
- UI/UX improvements
- Security auditing and improvements
- Performance optimizations
- Documentation improvements

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the Apache 2.0 license.

---

**Built with ❤️ for the digital credentialing and verifiable credentials community.**

*Empowering secure, verifiable digital achievements through cryptographic innovation.*