import nacl.signing
import nacl.encoding
import multibase
import json
import base64
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import secrets

def generate_key_id(issuer_id: str, key_index: int = 1) -> str:
    """
    Generate a unique key identifier for a verification method.
    
    Args:
        issuer_id (str): The issuer's DID or URL identifier
        key_index (int): Sequential number for the key (default: 1)
    
    Returns:
        str: Unique key identifier in the format "{issuer_id}#key-{key_index}"
    
    Example:
        >>> generate_key_id("https://example.edu/issuers/565049", 1)
        "https://example.edu/issuers/565049#key-1"
    """
    return f"{issuer_id}#key-{key_index}"

def create_ed25519_keypair() -> Tuple[str, str]:
    """
    Create a cryptographically secure Ed25519 key pair for signing credentials.
    
    Uses PyNaCl (libsodium) to generate a new random Ed25519 signing key pair
    and encodes both keys in multibase format using base58btc encoding.
    
    Returns:
        Tuple[str, str]: A tuple containing (public_key_multibase, private_key_multibase)
            - public_key_multibase: Base58btc-encoded public key with 'z' prefix
            - private_key_multibase: Base58btc-encoded private key with 'z' prefix
    
    Note:
        The private key should be stored securely and never exposed in logs or client-side code.
        The public key is embedded in the credential's verification method.
    """
    # Generate a new random signing key
    signing_key = nacl.signing.SigningKey.generate()
    
    # Get the verify (public) key
    verify_key = signing_key.verify_key
    
    # Convert to bytes
    private_key_bytes = signing_key.encode(encoder=nacl.encoding.RawEncoder)
    public_key_bytes = verify_key.encode(encoder=nacl.encoding.RawEncoder)
    
    # Convert to multibase format (base58btc encoding with 'z' prefix)
    private_key_multibase = multibase.encode('base58btc', private_key_bytes)
    public_key_multibase = multibase.encode('base58btc', public_key_bytes)
    
    return public_key_multibase, private_key_multibase

def create_signature(data_hash: str, private_key_multibase: str) -> str:
    """
    Create an Ed25519 digital signature for the provided data hash.
    
    Takes a SHA-256 hash of credential data and signs it using the provided
    Ed25519 private key, returning the signature in multibase format.
    
    Args:
        data_hash (str): Base64-encoded SHA-256 hash of the credential data
        private_key_multibase (str): Multibase-encoded (base58btc) Ed25519 private key
    
    Returns:
        str: Multibase-encoded (base58btc) Ed25519 signature with 'z' prefix
    
    Raises:
        ValueError: If the private key format is invalid
        nacl.exceptions.CryptoError: If signing fails
    
    Example:
        >>> data_hash = "SGVsbG8gV29ybGQ="  # Base64 encoded data
        >>> private_key = "z5TvdRmYr8U5..."  # Multibase encoded private key
        >>> signature = create_signature(data_hash, private_key)
        >>> signature.startswith('z')
        True
    """
    # Decode the private key from multibase format
    private_key_bytes = multibase.decode(private_key_multibase)
    
    # Create signing key from the private key
    signing_key = nacl.signing.SigningKey(private_key_bytes, encoder=nacl.encoding.RawEncoder)
    
    # Decode the hash from base64
    hash_bytes = base64.b64decode(data_hash)
    
    # Sign the hash
    signature_bytes = signing_key.sign(hash_bytes).signature
    
    # Encode the signature in multibase format
    signature_multibase = multibase.encode('base58btc', signature_bytes)
    
    return signature_multibase

def create_proof_hash(credential_data: Dict[str, Any]) -> str:
    """
    Create a SHA-256 hash of credential data for cryptographic proof generation.
    
    Removes any existing proof section from the credential, serializes the data
    to canonical JSON format, and creates a SHA-256 hash suitable for signing.
    
    Args:
        credential_data (Dict[str, Any]): The credential data dictionary to hash
    
    Returns:
        str: Base64-encoded SHA-256 hash of the canonical JSON representation
    
    Note:
        This function creates a deep copy of the credential data to avoid
        modifying the original dictionary when removing the proof section.
        The JSON serialization uses sorted keys and compact separators for
        canonical representation.
    
    Example:
        >>> credential = {"id": "urn:uuid:123", "type": ["VerifiableCredential"]}
        >>> hash_value = create_proof_hash(credential)
        >>> len(base64.b64decode(hash_value))
        32  # SHA-256 produces 32 bytes
    """
    # Remove existing proof if present
    credential_copy = credential_data.copy()
    credential_copy.pop("proof", None)
    
    # Serialize to canonical JSON
    canonical_json = json.dumps(credential_copy, sort_keys=True, separators=(',', ':'))
    
    # Create SHA-256 hash
    hash_bytes = hashlib.sha256(canonical_json.encode('utf-8')).digest()
    
    # Return base64 encoded hash
    return base64.b64encode(hash_bytes).decode('utf-8')

def get_current_timestamp() -> str:
    """
    Get the current UTC timestamp in ISO 8601 format.
    
    Returns:
        str: Current timestamp in ISO 8601 format with 'Z' suffix (e.g., "2025-07-15T14:30:00Z")
    
    Note:
        This function returns UTC time with the timezone indicator 'Z' instead of '+00:00'
        for compatibility with Open Badge 3.0 specifications and JSON-LD contexts.
    """
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def create_complete_signed_credential(
    credential_id: str,
    subject_id: str,
    issuer: Dict[str, Any],
    achievement: Dict[str, Any],
    issuer_key_id: str,
    private_key: str = None,
    valid_from: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a complete signed Open Badge 3.0 credential with cryptographic proof.
    
    This function generates a fully compliant Open Badge 3.0 credential including
    verification method, cryptographic proof, and all required metadata. If no
    private key is provided, a new Ed25519 key pair is generated.
    
    Args:
        credential_id (str): Unique identifier for the credential (e.g., "urn:uuid:...")
        subject_id (str): Identifier for the credential subject/recipient
        issuer (Dict[str, Any]): Issuer profile containing 'id', 'type', and 'name'
        achievement (Dict[str, Any]): Achievement definition with 'id', 'type', 'name', 
                                     'description', and 'criteria'
        issuer_key_id (str): Unique identifier for the issuer's verification key
        private_key (str, optional): Multibase-encoded Ed25519 private key for signing.
                                    If None, a new key pair is generated.
        valid_from (str, optional): ISO 8601 timestamp when credential becomes valid.
                                   If None, current timestamp is used.
        **kwargs: Additional arguments passed to build_openbadge_metadata()
    
    Returns:
        Dict[str, Any]: Complete signed credential dictionary including:
            - Standard Open Badge 3.0 fields (@context, id, type, issuer, etc.)
            - verificationMethod array with Ed25519 public key
            - proof section with Ed25519 signature
    
    Raises:
        ImportError: If required modules are not available
        ValueError: If required parameters are missing or invalid
        
    Example:
        >>> issuer = {
        ...     "id": "https://example.edu/issuers/565049",
        ...     "type": "Profile",
        ...     "name": "Example University"
        ... }
        >>> achievement = {
        ...     "id": "https://example.edu/achievements/badge",
        ...     "type": ["Achievement"],
        ...     "name": "Test Badge",
        ...     "description": "A test achievement",
        ...     "criteria": {"narrative": "Complete the test"}
        ... }
        >>> credential = create_complete_signed_credential(
        ...     credential_id="urn:uuid:123",
        ...     subject_id="did:example:subject",
        ...     issuer=issuer,
        ...     achievement=achievement,
        ...     issuer_key_id="https://example.edu/issuers/565049#key-1"
        ... )
        >>> "proof" in credential
        True
    """
    from modules.build_openbadge_metadata import build_openbadge_metadata, build_verification_method, build_proof_section
    
    # Generate keypair if private key not provided
    if not private_key:
        public_key, private_key = create_ed25519_keypair()
    else:
        public_key = derive_public_key_from_private(private_key)
    
    # Create verification method
    verification_method = build_verification_method(
        key_id=issuer_key_id,
        key_type="Ed25519VerificationKey2020",
        controller=issuer["id"],
        public_key_multibase=public_key
    )
    
    # Create initial credential without proof
    current_time = get_current_timestamp()
    credential_json = build_openbadge_metadata(
        credential_id=credential_id,
        subject_id=subject_id,
        issuer=issuer,
        valid_from=valid_from or current_time,
        achievement=achievement,
        verification_method=verification_method,
        **kwargs
    )
    
    # Parse credential for proof generation
    credential_data = json.loads(credential_json)
    
    # Create proof hash
    data_hash = create_proof_hash(credential_data)
    
    # Create cryptographic signature
    signature = create_signature(data_hash, private_key)
    
    # Create proof section
    proof = build_proof_section(
        proof_type="Ed25519Signature2020",
        created=current_time,
        verification_method=issuer_key_id,
        proof_purpose="assertionMethod",
        proof_value=signature
    )
    
    # Add proof to credential
    credential_data["proof"] = proof
    
    return credential_data

def derive_public_key_from_private(private_key_multibase: str) -> str:
    """
    Derive the public key from an Ed25519 private key in multibase format.
    
    Args:
        private_key_multibase (str): Multibase-encoded (base58btc) Ed25519 private key
    
    Returns:
        str: Multibase-encoded (base58btc) Ed25519 public key with 'z' prefix
    
    Raises:
        ValueError: If the private key format is invalid
        nacl.exceptions.CryptoError: If deriving the public key fails
    
    Example:
        >>> private_key = "z5TvdRmYr8U5..."  # Multibase encoded private key
        >>> public_key = derive_public_key_from_private(private_key)
        >>> public_key.startswith('z')
        True
    """
    # Decode the private key from multibase format
    private_key_bytes = multibase.decode(private_key_multibase)
    
    # Create signing key from the private key
    signing_key = nacl.signing.SigningKey(private_key_bytes, encoder=nacl.encoding.RawEncoder)
    
    # Get the verify (public) key
    verify_key = signing_key.verify_key
    
    # Convert to bytes
    public_key_bytes = verify_key.encode(encoder=nacl.encoding.RawEncoder)
    
    # Convert to multibase format (base58btc encoding with 'z' prefix)
    public_key_multibase = multibase.encode('base58btc', public_key_bytes)
    
    return public_key_multibase

# Example usage constants for testing
SAMPLE_ISSUER = {
    "id": "https://example.edu/issuers/565049",
    "type": "Profile",
    "name": "Example University"
}

SAMPLE_ACHIEVEMENT = {
    "id": "https://example.edu/achievements/robotics-badge",
    "type": ["Achievement"],
    "name": "Robotics Badge",
    "description": "Awarded for successfully building and programming a robot",
    "criteria": {
        "narrative": "Student must build and demonstrate a functional robot that can complete basic navigation tasks"
    }
}

if __name__ == "__main__":
    # Example usage
    credential = create_complete_signed_credential(
        credential_id="urn:uuid:12345678-1234-5678-1234-567812345678",
        subject_id="did:example:subject123",
        issuer=SAMPLE_ISSUER,
        achievement=SAMPLE_ACHIEVEMENT,
        issuer_key_id=generate_key_id(SAMPLE_ISSUER["id"], 1)
    )
    print(json.dumps(credential, indent=2))