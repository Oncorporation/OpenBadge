import json
from typing import Optional, Dict, List, Any

# Global constant for credentialSchema
CREDENTIAL_SCHEMA = {
    "id": "https://purl.imsglobal.org/spec/ob/v3p0/schema/json/ob_v3p0_achievementcredential_schema.json",
    "type": "JsonSchemaValidator2019"
}

def build_verification_method(
    key_id: str,
    key_type: str = "JsonWebKey2020",
    controller: str = None,
    public_key_jwk: Dict[str, Any] = None,
    public_key_multibase: str = None
) -> Dict[str, Any]:
    """
    Builds a verification method for Open Badge 3.0 credentials.
    
    Args:
        key_id (str): Unique identifier for the verification method
        key_type (str): Type of verification method (JsonWebKey2020, Ed25519VerificationKey2020, etc.)
        controller (str): The entity that controls this verification method
        public_key_jwk (Dict): JSON Web Key format public key
        public_key_multibase (str): Multibase encoded public key
    
    Returns:
        Dict: Verification method object
    """
    verification_method = {
        "id": key_id,
        "type": key_type,
        "controller": controller or key_id.split("#")[0]
    }
    
    if public_key_jwk:
        verification_method["publicKeyJwk"] = public_key_jwk
    elif public_key_multibase:
        # Ensure publicKeyMultibase is a string for JSON serialization
        if isinstance(public_key_multibase, bytes):
            verification_method["publicKeyMultibase"] = public_key_multibase.decode('utf-8')
        else:
            verification_method["publicKeyMultibase"] = public_key_multibase
    
    return verification_method

def build_proof_section(
    proof_type: str = "DataIntegrityProof",
    created: str = None,
    verification_method: str = None,
    proof_purpose: str = "assertionMethod",
    proof_value: str = None,
    cryptosuite: str = "eddsa-jcs-2022"
) -> Dict[str, Any]:
    """
    Builds a proof section for Open Badge 3.0 credentials.
    
    Args:
        proof_type (str): Type of proof (DataIntegrityProof, Ed25519Signature2020, etc.)
        created (str): ISO 8601 timestamp when proof was created
        verification_method (str): Reference to the verification method used
        proof_purpose (str): Purpose of the proof (assertionMethod, authentication, etc.)
        proof_value (str): The actual cryptographic proof value
        cryptosuite (str): Cryptographic suite used for the proof
    
    Returns:
        Dict: Proof section object
    """
    proof = {
        "type": proof_type,
        "created": created,
        "verificationMethod": verification_method,
        "proofPurpose": proof_purpose,
        "cryptosuite": cryptosuite
    }
    
    if proof_value:
        # Ensure proofValue is a string for JSON serialization
        if isinstance(proof_value, bytes):
            proof["proofValue"] = proof_value.decode('utf-8')
        else:
            proof["proofValue"] = proof_value
    
    return proof

def build_openbadge_metadata(
    credential_id: str,
    subject_id: str,
    issuer: dict,
    valid_from: str,
    achievement: dict,
    alignments: list[dict] = None,
    name: str = None,
    description: str = None,
    image: dict = None,
    verification_method: Dict[str, Any] = None,
    proof: Dict[str, Any] = None
) -> str:
    """
    Builds Open Badge 3.0 metadata as a JSON-LD string, incorporating required fields, 
    optional skills alignment, and cryptographic proof capabilities.

    Args:
        credential_id (str): Unique URI for the credential (e.g., 'urn:uuid:1234').
        subject_id (str): Identifier for the recipient (e.g., DID, email URI).
        issuer (dict): Issuer profile with 'id', 'type', 'name' (e.g., {'id': 'https://example.edu/issuers/565049', 'type': 'Profile', 'name': 'Example University'}).
        valid_from (str): ISO 8601 date when the credential is valid (e.g., '2025-06-25T09:07:00Z').
        achievement (dict): Achievement details, must include 'id', 'type' (['Achievement']), 'name', 'description', 'criteria'.
        alignments (list[dict], optional): List of skill alignments, each with 'targetName', 'targetUrl', etc.
        name (str, optional): Credential name for display.
        description (str, optional): Brief credential description.
        image (dict, optional): Image metadata (e.g., {'id': 'https://example.edu/images/badge.png', 'type': 'Image'}).
        verification_method (Dict, optional): Verification method for cryptographic proof.
        proof (Dict, optional): Cryptographic proof section.

    Returns:
        str: JSON-LD string representing the Open Badge 3.0 metadata.

    Example:
        issuer = {
            "id": "https://example.edu/issuers/565049",
            "type": "Profile",
            "name": "Example University"
        }
        achievement = {
            "id": "https://example.edu/achievements/123",
            "type": ["Achievement"],
            "name": "Robotics Badge",
            "description": "Awarded for building a robot",
            "criteria": {"narrative": "Build and demonstrate a robot"}
        }
        
        # Example verification method using Ed25519
        verification_method = {
            "id": "https://example.edu/issuers/565049#key-1",
            "type": "Ed25519VerificationKey2020",
            "controller": "https://example.edu/issuers/565049",
            "publicKeyMultibase": "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        }
        
        # Example proof section
        proof = {
            "type": "Ed25519Signature2020",
            "created": "2025-07-15T09:07:00Z",
            "verificationMethod": "https://example.edu/issuers/565049#key-1",
            "proofPurpose": "assertionMethod",
            "proofValue": "z3FXQjecWufY3QVes9kF6EEv6YfxJhsq2rjkRbQjABzrZcLZJFVQsN8CaQnwXBzSdW9LNVBcxCZmD8fmwqMgJ9VY"
        }
        
        metadata = build_openbadge_metadata(
            credential_id="urn:uuid:1234",
            subject_id="did:example:recipient",
            issuer=issuer,
            valid_from="2025-06-25T09:07:00Z",
            achievement=achievement,
            verification_method=verification_method,
            proof=proof
        )
    """
    # Set default type for the credential
    credential_type = ["VerifiableCredential", "OpenBadgeCredential"]

    # Construct credentialSubject
    credential_subject = {
        "type": ["AchievementSubject"],
        "id": subject_id,
        "achievement": {
            **achievement,
            "alignment": alignments if alignments else []
        }
    }

    # Construct the credential
    credential = {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.6.json"
        ],
        "id": credential_id,
        "type": credential_type,
        "issuer": issuer,
        "validFrom": valid_from,
        "credentialSubject": credential_subject,
        "credentialSchema": [CREDENTIAL_SCHEMA]
    }

    # Add optional fields if provided
    if name:
        credential["name"] = name
    if description:
        credential["description"] = description
    if image:
        credential["image"] = image
    
    # Add verification method if provided
    if verification_method:
        # Ensure verification method is properly serializable
        vm_copy = verification_method.copy()
        if "publicKeyMultibase" in vm_copy and isinstance(vm_copy["publicKeyMultibase"], bytes):
            vm_copy["publicKeyMultibase"] = vm_copy["publicKeyMultibase"].decode('utf-8')
        credential["verificationMethod"] = [vm_copy]
    
    # Add proof if provided
    if proof:
        # Ensure proof is properly serializable
        proof_copy = proof.copy()
        if "proofValue" in proof_copy and isinstance(proof_copy["proofValue"], bytes):
            proof_copy["proofValue"] = proof_copy["proofValue"].decode('utf-8')
        credential["proof"] = proof_copy

    # Serialize to JSON-LD string
    return json.dumps(credential, indent=2)

def create_sample_verification_methods():
    """
    Returns sample verification methods for different cryptographic algorithms
    as specified in the Open Badge 3.0 implementation guide.
    """
    
    # Ed25519 verification method (recommended for new implementations)
    ed25519_method = build_verification_method(
        key_id="https://example.edu/issuers/565049#key-1",
        key_type="Ed25519VerificationKey2020",
        controller="https://example.edu/issuers/565049",
        public_key_multibase="z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
    )
    
    # RSA verification method using JSON Web Key format
    rsa_jwk_method = build_verification_method(
        key_id="https://example.edu/issuers/565049#key-2",
        key_type="JsonWebKey2020",
        controller="https://example.edu/issuers/565049",
        public_key_jwk={
            "kty": "RSA",
            "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
            "e": "AQAB",
            "alg": "RS256",
            "use": "sig"
        }
    )
    
    # secp256k1 verification method for blockchain compatibility
    secp256k1_method = build_verification_method(
        key_id="https://example.edu/issuers/565049#key-3",
        key_type="JsonWebKey2020",
        controller="https://example.edu/issuers/565049",
        public_key_jwk={
            "kty": "EC",
            "crv": "secp256k1",
            "x": "WKn-ZIGevcwGIyyrzFoZNBdaq9_TsqzGHwHitJBcBmXQ",
            "y": "y77As5vbkazuEiiX9-fp5jhLLLMOoTUGrQZV5YJeJRZA",
            "alg": "ES256K",
            "use": "sig"
        }
    )
    
    return {
        "ed25519": ed25519_method,
        "rsa": rsa_jwk_method,
        "secp256k1": secp256k1_method
    }