from modules.build_openbadge_metadata import build_openbadge_metadata, build_verification_method, build_proof_section
from modules.crypto_utils import create_complete_signed_credential, SAMPLE_ISSUER, SAMPLE_ACHIEVEMENT
import json

def create_example_signed_badge():
    """
    Create an example signed Open Badge 3.0 credential with verification method and proof.
    """
    
    # Create a complete signed credential
    signed_credential = create_complete_signed_credential(
        credential_id="urn:uuid:91537dce-87d5-4277-8f30-f96b6a2e0af5",
        subject_id="did:example:ebfeb1f712ebc6f1c276e12ec21",
        issuer=SAMPLE_ISSUER,
        achievement=SAMPLE_ACHIEVEMENT,
        issuer_key_id="https://example.edu/issuers/565049#key-1",
        private_key="z3u2MQhLnQw7nvJRGJCdKdqfXHV4N7BLKuEGFWnJqsVSdgYv",
        name="Robotics Credential",
        description="Digital credential for robotics achievement"
    )
    
    return signed_credential

def demonstrate_verification_methods():
    """
    Demonstrate different types of verification methods supported by Open Badge 3.0.
    """
    from modules.build_openbadge_metadata import create_sample_verification_methods
    
    methods = create_sample_verification_methods()
    
    print("Available Verification Methods:")
    for method_type, method_data in methods.items():
        print(f"\n{method_type.upper()} Method:")
        print(json.dumps(method_data, indent=2))

if __name__ == "__main__":
    # Create example signed badge
    badge = create_example_signed_badge()
    
    print("Example Signed Open Badge 3.0 Credential:")
    print(json.dumps(badge, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Demonstrate verification methods
    demonstrate_verification_methods()